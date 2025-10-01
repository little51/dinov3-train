import torch
import torch.nn as nn
import timm
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

# COCO类别标签（80个类别 + 背景）
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO原始ID到连续ID的映射（1-90 -> 1-80）
COCO_ID_TO_CONTINUOUS_ID = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20,
    22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
    35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
    46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
    56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
    67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
    80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
}

# 连续ID到COCO原始ID的映射
CONTINUOUS_ID_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_CONTINUOUS_ID.items()}

# 模型输出ID到COCO连续ID的映射（91个输出 -> 80个实际类别）
# 模型输出索引0: 背景
# 模型输出索引1-90: 对应COCO ID 1-90，但只有80个有效类别
MODEL_OUTPUT_TO_COCO_ID = {}
MODEL_OUTPUT_TO_COCO_ID[0] = 0  # 背景

# 填充有效的COCO类别
for coco_id, continuous_id in COCO_ID_TO_CONTINUOUS_ID.items():
    MODEL_OUTPUT_TO_COCO_ID[coco_id] = continuous_id

# 对于不存在的COCO ID（12, 26, 29, 30, 45, 66, 68, 69, 71, 83），映射到背景
non_existent_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
for invalid_id in non_existent_ids:
    MODEL_OUTPUT_TO_COCO_ID[invalid_id] = 0  # 映射到背景

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

        # 检测头 - 保持91个类别的输出
        self.detection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes * 5)  # 91个类别 * 5个值
        )

    def forward(self, x):
        features = self.backbone(x)
        detection_output = self.detection_head(features)
        return detection_output


def load_model(model_path):
    """加载训练好的模型"""
    model = DINOv3DetectionModel(
        backbone_name='vit_large_patch16_dinov3.lvd1689m',
        num_classes=91  # 保持91个类别
    )

    # 加载训练好的检测头权重
    model.detection_head.load_state_dict(
        torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"模型加载完成: {model_path}")
    return model


def preprocess_image(image_path):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # 预处理后的图像用于模型推理
    processed_image = transform(image).unsqueeze(0)  # 添加batch维度

    return processed_image, image, original_size


def postprocess_detections(predictions, original_size, confidence_threshold=0.5):
    """后处理模型输出，提取检测结果"""
    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, 91, 5)  # [batch, 91个类别, 5]

    detections = []

    # 对每个模型输出类别进行处理
    for model_output_id in range(91):  # 0-90
        # [x, y, w, h, confidence]
        class_predictions = predictions[0, model_output_id]

        confidence = class_predictions[4].item()

        if confidence > confidence_threshold:
            # 映射到实际的COCO类别
            continuous_id = MODEL_OUTPUT_TO_COCO_ID.get(model_output_id, 0)
            
            # 如果是背景或无效类别，跳过
            if continuous_id == 0:
                continue
                
            # 获取COCO原始ID
            coco_id = CONTINUOUS_ID_TO_COCO_ID.get(continuous_id, 0)
            if coco_id == 0:
                continue

            # 提取边界框坐标 (归一化坐标)
            x, y, w, h = class_predictions[:4].cpu().detach().numpy()

            # 将归一化坐标转换回原图尺寸
            img_width, img_height = original_size
            x = x * img_width
            y = y * img_height
            w = w * img_width
            h = h * img_height

            # 计算边界框的左上角和右下角坐标
            x1 = max(0, x - w/2)
            y1 = max(0, y - h/2)
            x2 = min(img_width, x + w/2)
            y2 = min(img_height, y + h/2)

            # 确保边界框有效
            if x2 <= x1 or y2 <= y1:
                continue
                
            detection = {
                'class_id': coco_id,  # 使用COCO原始ID
                'continuous_id': continuous_id,  # 连续ID
                'class_name': COCO_CLASSES[continuous_id],  # 使用连续ID获取类别名
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'model_output_id': model_output_id  # 模型原始输出ID
            }
            detections.append(detection)

    # 按置信度排序
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    return detections


def visualize_detections(image, detections, output_path=None):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # 显示原图
    ax.imshow(image)

    # 为每个检测结果绘制边界框和标签
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']

        # 绘制矩形框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # 添加标签
        label = f'{class_name}: {confidence:.2f}'
        ax.text(
            bbox[0],
            bbox[1] - 10,
            label,
            color='red',
            fontsize=12,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5)
        )

    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
        print(f"结果已保存到: {output_path}")

    plt.show()


def detect_objects(image_path, model_path, confidence_threshold=0.5, output_path=None):
    """完整的目标检测流程"""

    # 1. 加载模型
    model = load_model(model_path)

    # 2. 预处理图像
    processed_image, original_image, original_size = preprocess_image(
        image_path)
    processed_image = processed_image.to(device)

    # 3. 模型推理
    with torch.no_grad():
        predictions = model(processed_image)
    
    # 4. 后处理
    detections = postprocess_detections(
        predictions, original_size, confidence_threshold)

    # 5. 打印结果
    print(f"检测到 {len(detections)} 个目标:")
    for i, detection in enumerate(detections):
        print(
            f"{i+1}. {detection['class_name']} (COCO ID: {detection['class_id']}): {detection['confidence']:.3f}")

    # 6. 可视化结果
    visualize_detections(original_image, detections, output_path)

    return detections


# 使用示例
if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'final_coco_detection_head.pth'  # 训练好的检测头权重文件
    IMAGE_PATH = 'test06.png'  # 要检测的图像路径
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
    OUTPUT_PATH = 'detection_result.jpg'  # 输出结果图像路径
    
    # 执行目标检测
    detections = detect_objects(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        output_path=OUTPUT_PATH
    )