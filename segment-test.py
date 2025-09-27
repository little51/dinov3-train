import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型类
class DINOv3SegmentationModel(nn.Module):
    """基于DINOv3的语义分割模型"""
    def __init__(self, backbone_name='vit_large_patch16_dinov3.sat493m', num_classes=10, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16  # DINOv3的patch大小
        
        # 加载预训练的DINOv3骨干网络
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # 获取特征，不包含分类头
            dynamic_img_size=True,
            img_size=img_size
        )
        
        # 冻结骨干网络所有参数
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

def load_model():
    """加载训练好的模型"""
    model = DINOv3SegmentationModel(
        backbone_name='vit_large_patch16_dinov3.sat493m',
        num_classes=10,
        img_size=224
    )
    checkpoint = torch.load('final_dino_segmentation_head.pth', map_location='cpu')
    model.head.load_state_dict(checkpoint['head_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_and_preprocess_image(image_path):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    print(f" 加载图像: {image_path} ({image.size[0]}x{image.size[1]})")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143)
        )
    ])
    
    return transform(image).unsqueeze(0).to(device), image

def segment_image(model, image_tensor):
    """执行语义分割"""
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1)[0]
    return prediction.cpu().numpy()

def analyze_land_use(segmentation_map):
    """分析土地利用情况"""
    class_names = [
        '一年生作物', '森林', '草本植被', '高速公路', 
        '工业区', '牧场', '多年生作物', '住宅区', 
        '河流', '海洋湖泊'
    ]
    
    total_pixels = segmentation_map.size
    results = []
    
    for class_id, class_name in enumerate(class_names):
        pixel_count = np.sum(segmentation_map == class_id)
        percentage = (pixel_count / total_pixels) * 100
        
        if percentage > 0.1:  # 只显示占比超过0.1%的类别
            results.append({
                '类别ID': class_id,
                '地物类型': class_name,
                '像素数量': pixel_count,
                '面积占比': percentage
            })
    
    # 按面积占比排序
    return sorted(results, key=lambda x: x['面积占比'], reverse=True)

def segment(file_path):
    # 1. 加载图像
    image_tensor,_ = load_and_preprocess_image(file_path)
    # 2. 执行分割
    segmentation_map = segment_image(model, image_tensor)
    print(" 语义分割完成")
    # 3. 分析土地利用
    land_use_results = analyze_land_use(segmentation_map)
    # 4. 显示结果
    print("\n 土地利用分析结果:")
    print("-" * 50)
    print(f"{'地物类型':<10} {'面积占比':<10} {'像素数量':<10}")
    print("-" * 50)
    total_percentage = 0
    for result in land_use_results:
        print(f"{result['地物类型']:<10} {result['面积占比']:<10.1f}% {result['像素数量']:<10}")
        total_percentage += result['面积占比']
    
    print("-" * 50)
    print(f"{'总计':<10} {total_percentage:<10.1f}% {segmentation_map.size:<10}")
    
    # 显示主要地物类型
    if land_use_results:
        main_landuse = land_use_results[0]
        print(f"\n 主要地物类型: {main_landuse['地物类型']} ({main_landuse['面积占比']:.1f}%)")
    print("\n===================================\n")

if __name__ == "__main__":
    model = load_model()
    segment("test04.png")
    segment("test05.png")