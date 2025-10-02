import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    # 加载模型，并设置 num_classes=0 以获取特征维度
    model = timm.create_model(
        'vit_small_patch16_dinov3.lvd1689m',
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
    # 根据自定义数据集的类别数设置
    num_classes = 3  # class1, class2, class3
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
    train_transforms = timm.data.create_transform(
        **data_config, is_training=True)

    # 验证和测试时使用相同的预处理，但不包含数据增强
    val_transforms = timm.data.create_transform(
        **data_config, is_training=False)

    # 加载自定义数据集
    data_dir = './mydataset'

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_transforms
    )

    # 由于数据集很小，使用较小的batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=1)
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=1)

    print(f"训练集类别: {train_dataset.classes}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print("装载数据集完成")

    return train_loader, val_loader, test_loader, data_config


def train_model(custom_model, train_loader, val_loader):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 只对分类头的参数进行优化
    optimizer = optim.Adam(custom_model.head.parameters(), lr=0.001)
    num_epochs = 5

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        custom_model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # 验证阶段
        val_acc = evaluate_model(custom_model, val_loader, "验证集")

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {running_loss/len(train_loader):.4f}, '
              f'训练准确率: {train_acc:.2f}%, '
              f'验证准确率: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(custom_model, 3, custom_model.backbone.num_features,
                       {'input_size': data_config['input_size']})
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    print('模型训练完成')


def evaluate_model(custom_model, data_loader, dataset_name="数据集"):
    correct = 0
    total = 0
    custom_model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{dataset_name}准确率: {accuracy:.2f}%')
    return accuracy


def save_model(custom_model, num_classes, feature_dim, data_config):
    # 只保存分类头的权重
    torch.save({
        'classifier_state_dict': custom_model.head.state_dict(),
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'training_config': {
            'model_name': 'vit_small_patch16_dinov3.lvd1689m',
            'input_size': data_config['input_size']
        }
    }, 'dino_classifier_head.pth')
    print("模型权重保存完成")


if __name__ == '__main__':
    model = load_model()
    custom_model, num_classes, feature_dim = def_model(model)
    train_loader, val_loader, test_loader, data_config = load_dataset(model)
    train_model(custom_model, train_loader, val_loader)

    # 最终在测试集上评估
    print("\n最终测试结果:")
    evaluate_model(custom_model, test_loader, "测试集")
