import torch
from sklearn.metrics import classification_report
from data_loader import get_data_loaders
from simple_model import SimpleCNN
import pandas as pd


def evaluate_model(data_dir, num_classes):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/custom/custom_model.pt"))
    model.eval()

    # Load validation dataset
    _, valid_loader = get_data_loaders(data_dir, batch_size=32)
    all_preds, all_labels = [], []

    # Perform evaluation
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Match target_names to folder names (all lowercase)
    target_names = ["abrasions", "bruises", "burns", "cut", "laseration", "stab_wound"]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    print("Classification Report:\n")
    print(pd.DataFrame(report).transpose())

    # Save report to CSV
    report_path = "QuickAid_validation_results.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Validation results saved to {report_path}")

if __name__ == "__main__":
    evaluate_model(data_dir="data/images", num_classes=6)
