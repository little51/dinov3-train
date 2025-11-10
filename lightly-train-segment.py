import lightly_train
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks

# 加载模型和预测
image_path = "test08.jpg"
model = lightly_train.load_model_from_checkpoint(
    "weights/lightlytrain_dinov3_eomt_vits16_cocostuff.pt"
)
# 处理掩码
masks = model.predict(image_path)
image = read_image(image_path)
masks = torch.stack([masks == class_id for class_id in masks.unique()])
image_with_masks = draw_segmentation_masks(image, masks, alpha=0.6)
# 显示图像
plt.figure(figsize=(12, 8))
plt.imshow(image_with_masks.permute(1, 2, 0))
plt.axis('off')
plt.title('result', fontsize=14)
plt.tight_layout()
plt.show()
