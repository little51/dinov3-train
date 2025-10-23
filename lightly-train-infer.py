import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torchvision import models
from torch import nn

def load_model(model_path, device='cpu'):
    """加载模型和类别映射"""
    checkpoint = torch.load(model_path, map_location=device)
    # 创建模型
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['class_to_idx']

def predict_image(image_path, model, class_to_idx, device='cpu'):
    """预测单张图像"""
    # 创建索引到类名的反向映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = idx_to_class[predicted_idx]
    return predicted_class, confidence

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载模型
    model, class_to_idx = load_model("out/model_for_inference.pth", device)
    print("加载的类别映射:", class_to_idx)
    # 进行预测
    image_path = "test_image.jpg"
    class_name, confidence = predict_image(image_path, model, class_to_idx, device)
    print(f"预测结果: {class_name}, 置信度: {confidence:.4f}")