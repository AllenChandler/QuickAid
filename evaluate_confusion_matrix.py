import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add 'src' to Python path
sys.path.append("J:/projects/QuickAid/src")

# 1. Load test dataset
data_dir = "J:/projects/QuickAid/data/test"  # Update this if test set path is different
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Match the size used in training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Load trained model
from custom_model.simple_model import get_pretrained_model

num_classes = len(test_dataset.classes)  # Automatically gets the number of classes
model = get_pretrained_model(num_classes=num_classes)
model.load_state_dict(torch.load("models/custom/custom_model.pt"))  # Update if path is different
model.eval()

# 3. Predict on test dataset
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to('cpu')  # Ensure inputs are on the same device as the model
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = test_dataset.classes  # e.g., ["chickenpox", "eczema"]

# 5. Plot the confusion matrix with enhanced formatting
plt.figure(figsize=(8, 7))  # Adjust figure size
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 12, "weight": "bold"},  # Larger, bold annotations
    cbar_kws={"shrink": 0.8}  # Adjust color bar size
)
plt.xlabel("Predicted", fontsize=12, fontweight="bold")  # Enhance axis labels
plt.ylabel("Actual", fontsize=12, fontweight="bold")
plt.title("Confusion Matrix", fontsize=16, fontweight="bold", loc="center")  # Bold, centered title
plt.xticks(fontsize=12)  # Adjust tick font size
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()  # Ensure everything fits nicely
plt.show()

# 6. Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)
