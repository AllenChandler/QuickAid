# QuickAid: Skin Disease Detection Using Deep Learning

## **Overview**
QuickAid is a machine learning project designed to classify images of skin diseases into one of the following categories:
- **ChickenPox**
- **Herpes**
- **Lupus**
- **Melanoma**
- **Monkeypox**
- **Sarampion**
- **Sarna**

Using a custom Convolutional Neural Network (CNN) built with PyTorch, the project aims to deliver high accuracy in diagnosing these conditions from image data.

## **Directory Structure**
```plaintext
QuickAid/
├── data/
│   ├── images/
│   │   ├── train/         # Training dataset (organized into subfolders by class)
│   │   ├── valid/         # Validation dataset (organized into subfolders by class)
│   │   └── test/          # Test dataset (organized into subfolders by class)
│   └── data_overview.txt  # Text file containing dataset stats
├── models/
│   ├── custom/
│   │   └── custom_model.pt  # Trained weights of the custom CNN
├── results/
│   ├── QuickAid_loss_log.xlsx  # Training and validation loss log
│   ├── QuickAid_loss_chart.png # Line chart of loss over epochs
│   └── QuickAid_test_results.csv # Test classification report
├── src/
│   ├── custom_model/
│   │   ├── data_loader.py    # Script for loading and transforming data
│   │   ├── simple_model.py   # Custom CNN model architecture
│   │   ├── train_custom.py   # Training script
│   │   └── evaluate_custom.py # Evaluation script for test data
│   └── evaluate.py           # Evaluation script for cross-validation
└── README.md                 # Project documentation
```

## **How It Works**
1. **Data Preparation:**
   - Organize images into train, validation, and test directories, with subfolders for each class.
   - Perform data augmentation (random rotations, flips, and cropping) during training to improve generalization.

2. **Training the Model:**
   - Train a simple CNN model with two convolutional layers followed by fully connected layers.
   - Save the model weights to `models/custom/custom_model.pt`.

3. **Evaluation:**
   - Evaluate the trained model on the test dataset.
   - Output a classification report and save the results to `QuickAid_test_results.csv`.

## **Dependencies**
To run the project, install the following Python libraries:
- `torch` and `torchvision`
- `sklearn`
- `pandas`
- `matplotlib`
- `openpyxl`

Install them using:
```bash
pip install torch torchvision scikit-learn pandas matplotlib openpyxl
```

## **Usage**
1. **Training the Model:**
   Run the training script to train the model:
   ```bash
   python src/custom_model/train_custom.py
   ```

2. **Evaluating the Model:**
   Evaluate the model using the test dataset:
   ```bash
   python src/custom_model/evaluate_custom.py
   ```
   The classification report will be saved in `results/QuickAid_test_results.csv`.

3. **Inspecting Results:**
   - Training and validation loss progression is logged in `QuickAid_loss_log.xlsx`.
   - A loss chart is saved as `QuickAid_loss_chart.png`.

## **Results**
The model achieved the following performance on the test dataset:
- **Accuracy:** 97.78%
- Detailed precision, recall, and F1-scores for each class are saved in `QuickAid_test_results.csv`.

## **Potential Improvements**
- Fine-tune the model with additional data augmentation or more advanced architectures.
- Experiment with different optimizers and learning rates.
- Test on real-world datasets to further validate the model.

## **Contact**
For questions or contributions, please contact the QuickAid team.

Author: Allen J Chandler
Email: allenjchandler@vt.edu
GitHub: [\[Your GitHub Profile URL\]](https://github.com/wonmorewave)

Author: Jordan A Holmes
Email: jah26603@vt.edu

---
**Disclaimer:** This project is for educational purposes only and should not be used as a diagnostic tool without further validation.

