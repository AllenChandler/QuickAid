# process_skin_sample.py

import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from custom_model.simple_model import get_pretrained_model

def get_most_recent_image(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    if not image_files:
        print("No images found in the skin_samples folder.")
        return None
    return max(image_files, key=os.path.getctime)

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_skin_sample():
    skin_samples_dir = os.path.abspath("../../skin_samples")
    model_path = os.path.abspath("../../models/custom/custom_model.pt")

    # Load class names from the saved file
    class_names_path = os.path.abspath("../../models/custom/class_names.txt")
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f]

    num_classes = len(class_names)

    image_path = get_most_recent_image(skin_samples_dir)
    if not image_path:
        return

    print(f"Processing image: {os.path.basename(image_path)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_model(num_classes=num_classes).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    input_image = process_image(image_path).to(device)

    try:
        with torch.no_grad():
            outputs = model(input_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            # Convert probabilities to numpy array for easier handling
            probabilities = probabilities.cpu().numpy()[0]

        # Print probabilities for each class
        print("\nClass Probabilities:")
        for cls, prob in zip(class_names, probabilities):
            print(f"{cls}: {prob:.4f}")

        if confidence < 0.6:
            print("\nPrediction: Uncertain")
        else:
            result = class_names[predicted_class]
            print(f"\nPredicted Disease: {result} (Confidence: {confidence:.2f})")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    classify_skin_sample()
