import matplotlib.pyplot as plt
import numpy as np

metrics = {
    "Precision": [0.88, 0.94],
    "Recall": [0.94, 0.86],
    "F1-Score": [0.91, 0.90],  
}
classes = ["Chickenpox", "Eczema"]

for metric, values in metrics.items():
    plt.figure(figsize=(16, 12))  # Increased figure size for better readability
    bars = plt.bar(classes, values, color=['skyblue', 'lightcoral'])
    plt.title(f"{metric} by Class", fontsize=32, fontweight='bold')  # Title font size doubled
    plt.xlabel("Classes", fontsize=28)  # X-axis label font size doubled
    plt.ylabel(metric, fontsize=28)  # Y-axis label font size doubled
    plt.ylim(0, 1.1)

    # Add text annotations at the top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + 0.01, 
            round(yval, 2), 
            va='bottom', 
            ha='center', 
            fontsize=24  # Annotation font size doubled
        )

    plt.xticks(fontsize=24)  # Class names font size doubled
    plt.yticks(fontsize=24)  # Y-axis ticks font size doubled
    plt.show()
