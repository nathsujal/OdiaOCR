import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

# ----- CONFIG -----
DATA_DIR = "segmented-characters"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LABEL ENCODING -----
char_labels = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.png')]
label_encoder = LabelEncoder()
label_encoder.fit(char_labels)
NUM_CLASSES = len(label_encoder.classes_)

# ----- CUSTOM DATASET -----
class OdiaCharacterDataset(Dataset):
    def __init__(self, data_dir, transform=None, multiply_factor=20):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transform
        self.multiply_factor = multiply_factor  # artificial data size boost

    def __len__(self):
        return len(self.filenames) * self.multiply_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.filenames)
        filename = self.filenames[real_idx]
        label_str = os.path.splitext(filename)[0]
        label = label_encoder.transform([label_str])[0]

        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

# ----- TRANSFORMS (with augmentation) -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- DATALOADER -----
dataset = OdiaCharacterDataset(DATA_DIR, transform=transform, multiply_factor=30)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- MODEL -----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x16
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(NUM_CLASSES).to(DEVICE)

# ----- TRAINING -----
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

# ----- SAVE MODEL -----
torch.save(model.state_dict(), "odia_char_ocr.pth")
print("✅ Model saved as 'odia_char_ocr.pth'")
