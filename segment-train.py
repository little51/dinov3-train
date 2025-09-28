import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("安全提醒：已全局禁用 SSL 证书验证")
except AttributeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子
torch.manual_seed(42)

class EuroSATSegmentationDataset(torch.utils.data.Dataset):
    """将EuroSAT分类数据集转换为伪分割数据集"""
    def __init__(self, eurosat_dataset, target_size=(224, 224)):
        self.dataset = eurosat_dataset
        self.target_size = target_size
        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                       'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                       'River', 'SeaLake']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 创建分割掩码：将整个图像标记为同一类别
        h, w = image.shape[1], image.shape[2]
        segmentation_mask = torch.full((h, w), label, dtype=torch.long)
        
        return image, segmentation_mask

class DINOv3SegmentationModel(nn.Module):
    """基于DINOv3的语义分割模型"""
    def __init__(self, backbone_name='vit_large_patch16_dinov3.sat493m', num_classes=10, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16  # DINOv3的patch大小
        
        # 加载预训练的DINOv3主干网络
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # 获取特征，不包含分类头
            dynamic_img_size=True,
            img_size=img_size
        )
        
        # 冻结主干网络所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 获取特征维度并计算特征图大小
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            self.feature_map_size = img_size // self.patch_size
            print(f"DINOv3特征维度: {self.feature_dim}")
            print(f"特征图大小: {self.feature_map_size}x{self.feature_map_size}")
        
        # 特征投影 - 将序列特征转换为2D特征图
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 分割解码器 (任务头)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # 上采样到原始尺寸
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 提取特征 - DINOv3输出的是序列化的patch特征
        # 形状: [batch_size, num_patches + 1, feature_dim] 或 [batch_size, num_patches, feature_dim]
        features = self.backbone(x)  # [B, N, C] 或 [B, C]
        
        # 处理特征形状
        if len(features.shape) == 2:
            # 如果输出是 [B, C]，需要重新构造为2D特征图
            num_patches = self.feature_map_size ** 2
            features = features.unsqueeze(1)  # [B, 1, C]
            features = features.expand(-1, num_patches, -1)  # [B, N, C]
        
        # 移除[CLS] token（如果存在），只保留patch tokens
        if features.shape[1] == self.feature_map_size ** 2 + 1:
            # 包含[CLS] token，移除它
            patch_features = features[:, 1:, :]  # [B, N, C]
        else:
            # 不包含[CLS] token
            patch_features = features  # [B, N, C]
        
        # 重塑为2D特征图 [B, H, W, C]
        h = w = self.feature_map_size
        features_2d = patch_features.view(batch_size, h, w, -1)  # [B, H, W, C]
        
        # 调整维度顺序 [B, C, H, W]
        features_2d = features_2d.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 特征投影
        features_2d = features_2d.contiguous()
        features_2d = self.feature_projection(features_2d.permute(0, 2, 3, 1))  # [B, H, W, projected_C]
        features_2d = features_2d.permute(0, 3, 1, 2)  # [B, projected_C, H, W]
        
        # 分割解码
        seg_output = self.head(features_2d)
        seg_output = self.upsample(seg_output)
        
        return seg_output

def load_segmentation_dataset(batch_size=8):
    """加载分割数据集"""
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143))
    ])
    
    # 加载EuroSAT数据集
    full_dataset = EuroSAT(root='./data', download=True, transform=train_transform)
    # 转换为分割数据集
    segmentation_dataset = EuroSATSegmentationDataset(full_dataset)
    # 分割数据集
    dataset_size = len(segmentation_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        segmentation_dataset, [train_size, val_size, test_size]
    )
    # 为验证集和测试集应用不同的变换
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"分割数据集加载完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader

def train_segmentation_model(model, train_loader, val_loader, num_epochs=50):
    """训练分割模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 优化器 - 使用较小的学习率
    optimizer = optim.AdamW(
        [
            {'params': model.backbone.parameters(), 'lr': 0.0001},  # 主干网络较小学习率
            {'params': model.feature_projection.parameters(), 'lr': 0.001},
            {'params': model.head.parameters(), 'lr': 0.001},    # 头正常学习率
        ],
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_mious = []
    best_miou = 0.0
    
    print("开始训练分割模型...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        miou = evaluate_segmentation(model, val_loader)
        val_mious.append(miou)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_loss:.4f}')
        print(f'  验证mIoU: {miou:.4f}')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型 - 只保存任务头
        if miou > best_miou:
            best_miou = miou
            torch.save(model.head.state_dict(), 'best_dino_segmentation_head.pth')
            print(f'  新的最佳模型已保存! mIoU: {best_miou:.4f}')
    
    print('分割模型训练完成!')
    return model, train_losses, val_mious

def evaluate_segmentation(model, data_loader):
    """评估分割模型性能"""
    model.eval()
    total_iou = 0.0
    num_batches = 0
    num_classes = 10
    
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算每个批次的mIoU
            batch_iou = calculate_iou(preds, masks, num_classes)
            total_iou += batch_iou
            num_batches += 1
    
    model.train()
    return total_iou / num_batches if num_batches > 0 else 0.0

def calculate_iou(preds, targets, num_classes):
    """计算交并比"""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))
    
    # 忽略NaN值计算平均值
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('./data', exist_ok=True)
    
    print("步骤 1/5: 创建DINOv3分割模型...")
    model = DINOv3SegmentationModel(
        backbone_name='vit_large_patch16_dinov3.sat493m',
        num_classes=10,
        img_size=224
    )
    
    print("步骤 2/5: 加载EuroSAT数据集...")
    train_loader, val_loader, test_loader = load_segmentation_dataset(batch_size=8)
    
    print("步骤 3/5: 训练模型...")
    trained_model, train_losses, val_mious = train_segmentation_model(
        model, train_loader, val_loader, num_epochs=5
    )
    
    print("步骤 4/5: 评估模型...")
    # 最终评估
    test_miou = evaluate_segmentation(trained_model, test_loader)
    print(f'测试集 mIoU: {test_miou:.4f}')
    
    print("步骤 5/5: 保存任务头...")
    # 保存最终模型 - 只保存任务头
    torch.save({
        'head_state_dict': trained_model.head.state_dict(),  # 只保存头
        'num_classes': 10,
        'model_config': {
            'backbone': 'vit_large_patch16_dinov3.sat493m',
            'input_size': (224, 224)
        }
    }, 'final_dino_segmentation_head.pth')
    
    print("DINOv3语义分割模型训练完成! 任务头已保存。")

if __name__ == '__main__':
    main()