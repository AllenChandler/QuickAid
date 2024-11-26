import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simple_model import SimpleCNN
from sklearn.metrics import classification_report
import pandas as pd


def evaluate_model(data_dir, model_path, num_classes, batch_size=32):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    class_names = test_dataset.classes
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("Classification Report:\n")
    print(pd.DataFrame(report).transpose())  # Print as a formatted table

    # Save report to a CSV file
    report_path = "QuickAid_test_results.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    evaluate_model(
        data_dir="data/images", 
        model_path="models/custom/custom_model.pt", 
        num_classes=7
    )
