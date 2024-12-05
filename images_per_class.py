import collections
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define a function to count images per class
def count_images(data_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_counts = collections.Counter(dataset.targets)
    classes = dataset.classes
    counts = [class_counts[i] for i in range(len(classes))]
    return classes, counts

# Directories for test and train datasets
test_data_dir = "J:/projects/QuickAid/data/test"  # Update if the path is different
train_data_dir = "J:/projects/QuickAid/data/train"  # Update if the path is different

# Count images for test and train datasets
test_classes, test_counts = count_images(test_data_dir)
train_classes, train_counts = count_images(train_data_dir)

# Combine all bars into one plot
all_classes = [
    f"Train-{cls}" for cls in train_classes
] + [
    f"Test-{cls}" for cls in test_classes
]
all_counts = train_counts + test_counts

# Plot the combined bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(all_classes, all_counts, color=['skyblue', 'lightcoral'] * 2)

# Add actual counts above each bar
for bar, count in zip(bars, all_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        count + 5,
        str(count),
        ha='center',
        fontsize=20,  # Doubled font size for the counts
        fontweight='bold'
    )

# Customize the chart
plt.title("Class Distribution - Train & Test Datasets", fontsize=32, fontweight='bold')  # Doubled title size
plt.xlabel("Classes", fontsize=28)  # Doubled axis label size
plt.ylabel("Number of Images", fontsize=28)
plt.xticks(rotation=45, fontsize=20)  # Doubled tick label size
plt.yticks(fontsize=20)
plt.tight_layout()

plt.show()
