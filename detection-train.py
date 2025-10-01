import torch
import torch.nn as nn
import timm
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class DINOv3DetectionModel(nn.Module):
    """基于DINOv3的目标检测模型"""

    def __init__(self, backbone_name='vit_large_patch16_dinov3.lvd1689m', num_classes=91):
        super().__init__()

        # 加载预训练的DINOv3主干网络
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )

        # 冻结主干网络
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            print(f"DINOv3特征维度: {self.feature_dim}")

        # 检测头 - 输出边界框坐标和类别概率
        self.detection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes * 5)  # 每个类别: 4个坐标(x,y,w,h) + 1个置信度
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, C]
        detection_output = self.detection_head(
            features)  # [B, num_classes * 5]
        return detection_output


def collate_fn(batch):
    """自定义collate函数处理COCO标注"""
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        # 简化处理：只取第一个目标的标注
        if len(target) > 0:
            bbox = target[0]['bbox']  # [x, y, width, height]
            category_id = target[0]['category_id']
            targets.append({
                'bbox': torch.tensor(bbox, dtype=torch.float32),
                'category_id': torch.tensor(category_id, dtype=torch.long)
            })
        else:
            # 如果没有目标，使用默认值
            targets.append({
                'bbox': torch.tensor([0, 0, 1, 1], dtype=torch.float32),
                'category_id': torch.tensor(0, dtype=torch.long)
            })

    return torch.stack(images), targets


def load_coco_dataset(batch_size=8):
    """加载COCO2017数据集"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    # 加载COCO数据集
    train_dataset = CocoDetection(
        root='./data/coco/train2017',
        annFile='./data/coco/annotations/instances_train2017.json',
        transform=transform
    )

    val_dataset = CocoDetection(
        root='./data/coco/val2017',
        annFile='./data/coco/annotations/instances_val2017.json',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    print(f"COCO数据集加载完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader


class DetectionLoss(nn.Module):
    """简化的检测损失函数"""

    def __init__(self, num_classes=91):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        predictions = predictions.view(batch_size, self.num_classes, 5)

        bbox_loss = 0
        cls_loss = 0

        for i, target in enumerate(targets):
            # 将目标数据移动到与预测相同的设备
            gt_bbox = target['bbox'].to(predictions.device)
            gt_cls = target['category_id'].to(predictions.device)
            
            # 边界框损失
            pred_bbox = predictions[i, target['category_id'], :4]  # 预测的边界框
            bbox_loss += self.bbox_loss(pred_bbox, gt_bbox)

            # 分类损失
            pred_cls = predictions[i, :, 4]  # 所有类别的置信度
            cls_loss += self.cls_loss(pred_cls.unsqueeze(0),
                                      gt_cls.unsqueeze(0))

        total_loss = (bbox_loss + cls_loss) / batch_size
        return total_loss


def train_detection_model(model, train_loader, val_loader, num_epochs=10):
    """训练检测模型"""
    model.to(device)
    criterion = DetectionLoss(num_classes=91)
    optimizer = optim.AdamW(
        model.detection_head.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("开始训练检测模型...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, targets in train_bar:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(
            f'Epoch {epoch+1}: 损失: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # 保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save(model.detection_head.state_dict(),
                       f'coco_detection_head_epoch{epoch+1}.pth')

    return model


def main():
    """主函数"""
    # 创建数据目录
    os.makedirs('./data/coco', exist_ok=True)

    print("步骤 1/4: 创建DINOv3检测模型...")
    model = DINOv3DetectionModel(
        backbone_name='vit_large_patch16_dinov3.lvd1689m',
        num_classes=91  # COCO有80个类别+背景
    )

    print("步骤 2/4: 加载COCO2017数据集...")
    train_loader, val_loader = load_coco_dataset(batch_size=8)

    print("步骤 3/4: 训练检测头...")
    trained_model = train_detection_model(
        model, train_loader, val_loader, num_epochs=20)

    print("步骤 4/4: 保存最终检测头...")
    torch.save(trained_model.detection_head.state_dict(),
               'final_coco_detection_head.pth')
    print("COCO目标检测头训练完成!")


if __name__ == '__main__':
    main()