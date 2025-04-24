import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import os

# ----- MODEL DEFINITION (same as training) -----
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

# ----- CHARACTER SEGMENTATION FUNCTIONS -----
def filter_overlapping_boxes(boxes, overlap_threshold=0.5):
    """Filter out boxes that are contained within larger boxes."""
    if not boxes:
        return []
    
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        is_contained = False
        for j, box2 in enumerate(boxes):
            if i != j:
                x1_1, y1_1, w1, h1 = box1
                x1_2, y1_2, w2, h2 = box2
                # Check if box1 is inside box2
                if (x1_1 >= x1_2 and y1_1 >= y1_2 and
                    x1_1 + w1 <= x1_2 + w2 and y1_1 + h1 <= y1_2 + h2):
                    is_contained = True
                    break
        if not is_contained:
            filtered_boxes.append(box1)
    return filtered_boxes

def merge_vertical_boxes(boxes, vertical_threshold=5, horizontal_tolerance=5):
    """Merge vertically aligned boxes with small vertical gaps."""
    if not boxes:
        return []
    
    # Sort boxes by y-coordinate (top to bottom)
    sorted_boxes = sorted(boxes, key=lambda box: box[1])
    
    merged_boxes = []
    current_group = [sorted_boxes[0]]
    
    for box in sorted_boxes[1:]:
        last_box = current_group[-1]
        vertical_distance = box[1] - (last_box[1] + last_box[3])
        
        # Check vertical proximity and horizontal alignment
        if vertical_distance <= vertical_threshold:
            x_diff = abs(box[0] - last_box[0])
            if x_diff <= horizontal_tolerance:
                current_group.append(box)
            else:
                merged_boxes.append(merge_group(current_group))
                current_group = [box]
        else:
            merged_boxes.append(merge_group(current_group))
            current_group = [box]
    
    merged_boxes.append(merge_group(current_group))
    return merged_boxes

def merge_group(group):
    """Merge a group of boxes into a single bounding box."""
    x = min(b[0] for b in group)
    y = min(b[1] for b in group)
    max_x = max(b[0] + b[2] for b in group)
    max_y = max(b[1] + b[3] for b in group)
    return (x, y, max_x - x, max_y - y)

def segment_and_predict(image_path, model, label_encoder, output_path=None):
    """Segment characters from image, predict them, and display/save results."""
    # Read image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Make a copy for visualization
    display_image = original_image.copy()
    
    # Preprocess for character segmentation
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get character bounding boxes
    contour_boxes = []
    min_area = 20  # Adjust based on your image resolution
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > min_area:
            contour_boxes.append((x, y, w, h))
    
    # Filter and merge boxes
    filtered_boxes = filter_overlapping_boxes(contour_boxes)
    merged_boxes = merge_vertical_boxes(filtered_boxes)
    
    # Determine average line height for line grouping
    if merged_boxes:
        line_height = sum(box[3] for box in merged_boxes) / len(merged_boxes) * 1.5
    else:
        line_height = 30  # Default if no boxes
    
    # Group boxes by lines based on their y-coordinate
    lines = {}
    for box in merged_boxes:
        x, y, w, h = box
        line_idx = int(y // line_height)  # Convert to integer for line index
        if line_idx not in lines:
            lines[line_idx] = []
        lines[line_idx].append(box)
    
    # Sort each line from left to right
    for line_idx in lines:
        lines[line_idx] = sorted(lines[line_idx], key=lambda box: box[0])
    
    # Flatten boxes in reading order (top to bottom, left to right)
    sorted_boxes = []
    for line_idx in sorted(lines.keys()):
        sorted_boxes.extend(lines[line_idx])
    
    # Prepare for prediction
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    model.eval()
    predictions = []
    
    # Process each character
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        # Extract character image
        char_img = gray[y:y+h, x:x+w]
        
        # Skip if the image is empty
        if char_img.size == 0:
            continue
        
        # Convert to PIL and preprocess
        pil_img = Image.fromarray(char_img)
        input_tensor = transform(pil_img).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = label_encoder.inverse_transform([predicted_idx.item()])[0]
        
        # Store prediction with box and line info
        line_idx = int(y // line_height)
        predictions.append((predicted_label, (x, y, w, h), line_idx))
        
        # Draw prediction on image
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_image, predicted_label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.title("Character Recognition Results", fontsize=16)
    plt.axis('off')
    
    # Save results to file if output path provided
    if output_path:
        # Group predictions by line
        line_predictions = {}
        for pred, box, line_idx in predictions:
            if line_idx not in line_predictions:
                line_predictions[line_idx] = []
            line_predictions[line_idx].append(pred)
        
        # Create formatted text in reading order
        text_lines = []
        for line_idx in sorted(line_predictions.keys()):
            text_lines.append("  ".join(line_predictions[line_idx]))
        
        formatted_text = "\n\n".join(text_lines)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        print(f"Predictions saved to {output_path}")
        
        # Save the visualization
        image_output = os.path.splitext(output_path)[0] + "_visualization.jpg"
        cv2.imwrite(image_output, display_image)
        print(f"Visualization saved to {image_output}")
    
    plt.show()
    return predictions

def main():
    # ----- CONFIGURATION -----
    MODEL_PATH = "app/odia_char_ocr.pth"
    DATA_DIR = "segmented-characters"  # For label encoder
    IMG_PATH = "/home/sujalnath/Pictures/Screenshots/Screenshot from 2025-04-24 16-37-22.png"  # Path to the image you want to process
    OUTPUT_PATH = "app/predictions.txt"  # Path to save the prediction text
    
    # ----- DEVICE -----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # ----- PREPARE LABEL ENCODER -----
    char_labels = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    label_encoder = LabelEncoder()
    label_encoder.fit(char_labels)
    NUM_CLASSES = len(label_encoder.classes_)
    print(f"Found {NUM_CLASSES} character classes")
    
    # ----- LOAD MODEL -----
    model = SimpleCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model loaded from {MODEL_PATH}")
    
    # ----- RUN PREDICTION -----
    segment_and_predict(IMG_PATH, model, label_encoder, OUTPUT_PATH)

if __name__ == "__main__":
    main()