import torch
import torch.nn as nn
import timm
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    # 加载模型，并设置 num_classes=0 以获取特征维度
    model = timm.create_model(
        'timm/vit_small_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0)
    model.eval()  # 设置为评估模式
    # 冻结模型的所有参数
    for param in model.parameters():
        param.requires_grad = False
    print("模型装载完成")
    return model


def def_model(model):
    # 获取DINOv3输出的特征维度
    feature_dim = model.num_features
    # 假设你的数据集有10个类别
    num_classes = 10
    # 定义一个简单的分类头
    classifier = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes)
    )
    # 将骨干网络和分类头组合成完整模型
    class CustomDINOv3(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)  # 形状为 [batch_size, feature_dim]
            output = self.head(features)
            return output
    custom_model = CustomDINOv3(model, classifier)
    print("定义训练模型完成")
    return custom_model, num_classes, feature_dim


def load_dataset(model):
    # 获取模型对应的数据预处理配置
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(
        **data_config, is_training=True)
    # 加载CIFAR-10数据集（示例）
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=2)
    print("装载数据集完成")
    return train_loader, test_loader, data_config


def train_model(custom_model, train_loader):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 只对分类头的参数进行优化
    optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        custom_model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('模型训练完成')


def save_model(custom_model, num_classes, feature_dim, data_config):
    # 只保存分类头的权重（文件更小，更灵活）
    torch.save({
        'classifier_state_dict': custom_model.head.state_dict(),
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'training_config': {
            'model_name': 'facebook/dinov3-base-pretrain-lvd1689m',
            'input_size': data_config['input_size']
        }
    }, 'dino_classifier_head.pth')
    print("模型权重保存完成")


def eval_mode(custom_model, test_loader):
    correct = 0
    total = 0
    custom_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试集准确率: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    model = load_model()
    custom_model, num_classes, feature_dim = def_model(model)
    train_loader, test_loader, data_config = load_dataset(model)
    train_model(custom_model, train_loader)
    save_model(custom_model, num_classes, feature_dim, data_config)
    eval_mode(custom_model, test_loader)
