import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# ----- CONFIG -----
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- LOAD LABEL ENCODER -----
# Recreate label encoder using same filenames
DATA_DIR = "segmented-characters"
char_labels = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.png')]
label_encoder = LabelEncoder()
label_encoder.fit(char_labels)

# ----- MODEL DEFINITION (must match training) -----
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

# ----- LOAD MODEL -----
NUM_CLASSES = len(label_encoder.classes_)
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("odia_char_ocr.pth", map_location=DEVICE))
model.eval()

# ----- TRANSFORMS -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- PREDICTION FUNCTION -----
def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        pred_index = torch.argmax(output, dim=1).item()
        pred_label = label_encoder.inverse_transform([pred_index])[0]

    return pred_label

# ----- USAGE -----
image_path = "segmented-characters/କ୍କ.png"  # Replace with your test image
predicted_char = predict_image(image_path)
print("🔤 Predicted Character:", predicted_char)
