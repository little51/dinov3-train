import torch
import torchvision.transforms.v2 as v2
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = datasets.ImageFolder(root="my_data_dir", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Load the exported model
model = models.resnet18()
model.load_state_dict(torch.load(
    "out/my_experiment/exported_models/exported_last.pt", weights_only=True))

# Update the classification head with the correct number of classes
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Starting fine-tuning...")
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx,
    'num_classes': len(dataset.classes),
    'epoch': num_epochs,
    'loss': avg_loss
}, f"out/model_for_inference.pth")