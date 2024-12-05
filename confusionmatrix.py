from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Chickenpox", "Eczema"])
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
