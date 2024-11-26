import torch
from sklearn.metrics import classification_report
from data_loader import get_data_loaders  # Adjusted import
from simple_model import SimpleCNN       # Adjusted import
import pandas as pd

def evaluate_model(data_dir, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/custom/custom_model.pt"))
    model.eval()

    _, valid_loader = get_data_loaders(data_dir, batch_size=32)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=["ChickenPox", "Herpes", "Lupus", "Melanoma", "Monkeypox", "Sarampion", "Sarna"]))

if __name__ == "__main__":
    evaluate_model(data_dir="data/images", num_classes=7)
