import torch
import torch.nn as nn
import timm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() :
    backbone = timm.create_model('timm/vit_small_patch16_dinov3.lvd1689m', 
                            pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(backbone)
    backbone.eval()
    # 加载分类头
    checkpoint = torch.load('dino_classifier_head.pth', map_location=device)
    classifier = nn.Sequential(
        nn.Linear(checkpoint['feature_dim'], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, checkpoint['num_classes'])
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    # 组合模型
    model = nn.Sequential(backbone, classifier)
    model.to(device)
    model.eval()
    return model,data_config

def process_image(data_config,file_path) :
    # 预处理和分类
    transform = timm.data.create_transform(**data_config, is_training=False)
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

def classifier(model,input_tensor) :
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"预测: {class_names[predicted.item()]} (置信度: {confidence.item():.4f})")

if __name__ == '__main__':
    model,data_config = load_model()
    input_tensor = process_image(data_config,'test01.png')
    classifier(model,input_tensor)
    input_tensor = process_image(data_config,'test02.png')
    classifier(model,input_tensor)
    input_tensor = process_image(data_config,'test03.png')
    classifier(model,input_tensor)