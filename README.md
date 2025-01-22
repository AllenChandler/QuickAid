# QuickAid: Skin Disease Classification Using Deep Learning

## Overview

**QuickAid** is a deep learning project aimed at classifying skin diseases, specifically distinguishing between **Chickenpox** and **Eczema** from images. By leveraging a fine-tuned ResNet-18 convolutional neural network, QuickAid aspires to assist in early diagnosis and treatment planning, making dermatological assessments more accessible and efficient.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [Data Collection](#data-collection)
  - [Directory Structure](#directory-structure)
- [Model Architecture](#model-architecture)
  - [ResNet-18 Overview](#resnet-18-overview)
  - [Fine-Tuning Strategy](#fine-tuning-strategy)
- [Training the Model](#training-the-model)
  - [Data Augmentation](#data-augmentation)
  - [Hyperparameters](#hyperparameters)
  - [Training Process](#training-process)
- [Evaluating the Model](#evaluating-the-model)
  - [Performance Metrics](#performance-metrics)
  - [Results](#results)
- [Classifying New Images](#classifying-new-images)
  - [Using the Command Line](#using-the-command-line)
  - [Interpreting the Output](#interpreting-the-output)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Potential Improvements](#potential-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Binary Classification**: Accurately classifies images as either Chickenpox or Eczema.
- **Transfer Learning**: Utilizes a pre-trained ResNet-18 model fine-tuned on custom data.
- **Data Augmentation**: Enhances model robustness through extensive data augmentation techniques.
- **Model Checkpointing**: Saves the best model during training based on validation loss.
- **Easy Deployment**: Scripts provided for training, evaluation, and real-time image classification.
- **Comprehensive Evaluation**: Generates detailed performance metrics and confusion matrices.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python 3.7 or higher**
- **Git** (for cloning the repository)
- **pip** (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/QuickAid.git
   cd QuickAid
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv env
   source env/bin/activate      # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install dependencies manually:

   ```bash
   pip install torch torchvision scikit-learn pandas matplotlib openpyxl Pillow
   ```

---

## Dataset Preparation

### Data Collection

The datasets were sourced from Kaggle:

- **Chickenpox Dataset**: [Skin Lesion Dataset Using Segmentation](https://www.kaggle.com/datasets/devdope/skin-lesion-dataset-using-segmentation)
- **Eczema Dataset**: [Balanced and Enhanced Skin Disease Dataset](https://www.kaggle.com/datasets/divyaiit/balanced-and-enhanced-skin-disease-dataset)

Download the datasets and extract the images.

### Directory Structure

Organize the datasets into the following directory structure:

```plaintext
QuickAid/
└── data/
    ├── train/
    │   ├── chickenpox/
    │   │   └── [chickenpox training images]
    │   └── eczema/
    │       └── [eczema training images]
    ├── valid/
    │   ├── chickenpox/
    │   │   └── [chickenpox validation images]
    │   └── eczema/
    │       └── [eczema validation images]
    └── test/
        ├── chickenpox/
        │   └── [chickenpox test images]
        └── eczema/
            └── [eczema test images]
```

- **Training Set**: Approximately 70% of the images.
- **Validation Set**: Approximately 15% of the images.
- **Test Set**: Approximately 15% of the images.

Ensure that each class has a balanced number of images in each set to prevent model bias.

---

## Model Architecture

### ResNet-18 Overview

ResNet-18 is a deep convolutional neural network known for its residual learning framework, which allows training of deeper networks without degradation.

- **Residual Blocks**: Helps in mitigating the vanishing gradient problem.
- **Pre-trained Weights**: Utilizes weights trained on the ImageNet dataset, providing a strong starting point.

### Fine-Tuning Strategy

- **Layer Freezing**: Initially, all layers except the final fully connected layer are frozen.
- **Selective Unfreezing**: Unfreeze the last few layers to allow the model to learn more specific features relevant to the skin disease classification task.
- **Output Layer Modification**: The final fully connected layer is replaced with a new layer that outputs two classes.

---

## Training the Model

### Data Augmentation

To improve model generalization, the following augmentation techniques are applied:

- **Random Resized Cropping**
- **Random Rotation (up to 45 degrees)**
- **Horizontal and Vertical Flipping**
- **Color Jitter (brightness, contrast, saturation, hue)**
- **Random Affine Transformations**

### Hyperparameters

- **Learning Rate**: Starts at `1e-4`, adjusted using a learning rate scheduler.
- **Batch Size**: 32
- **Optimizer**: AdamW with weight decay (`1e-5`)
- **Loss Function**: Cross-Entropy Loss
- **Number of Epochs**: 50 (with checkpointing)

### Training Process

Run the training script:

```bash
python src/custom_model/train_custom.py
```

- **Model Checkpointing**: The model saves weights to `models/custom/best_model.pt` when validation loss improves.
- **Logging**: Training and validation losses are logged and saved to `results/QuickAid_loss_log.xlsx`.
- **Loss Plot**: A plot of the training and validation loss over epochs is saved as `results/QuickAid_loss_chart.png`.

---

## Evaluating the Model

### Performance Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

### Results

Evaluate the model using:

```bash
python src/custom_model/evaluate_custom.py
```

- The classification report is saved to `results/QuickAid_test_results.csv`.
- Example of expected output:

```plaintext
Classification Report:

              precision    recall  f1-score   support

  Chickenpox       0.88      0.90      0.89       200
      Eczema       0.89      0.87      0.88       200

    accuracy                           0.88       400
   macro avg       0.88      0.88      0.88       400
weighted avg       0.88      0.88      0.88       400

Test Accuracy: 88.00%
```


---

## Classifying New Images

### Using the Command Line

Place the image you want to classify into the `skin_samples` directory.

Run the script:

```bash
python src/custom_model/process_skin_sample.py
```

### Interpreting the Output

The script will output:

- The predicted class (Chickenpox or Eczema)
- The confidence score
- Probabilities for each class

Example output:

```plaintext
Processing image: sample_image.jpg

Class Probabilities:
Chickenpox: 0.7624
Eczema: 0.2376

Predicted Disease: Chickenpox (Confidence: 76.24%)
```

---

## Project Structure

```plaintext
QuickAid/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   └── custom/
│       ├── custom_model.pt
│       ├── best_model.pt
│       └── class_names.txt
├── results/
│   ├── QuickAid_loss_log.xlsx
│   ├── QuickAid_loss_chart.png
│   └── QuickAid_test_results.csv
├── skin_samples/
│   └── [your sample images]
├── src/
│   ├── custom_model/
│   │   ├── data_loader.py
│   │   ├── simple_model.py
│   │   ├── train_custom.py
│   │   ├── evaluate_custom.py
│   │   └── process_skin_sample.py
│   └── README.md
├── requirements.txt
└── LICENSE
```

- **data/**: Contains the dataset organized for training, validation, and testing.
- **models/**: Stores the trained model weights and class names.
- **results/**: Outputs from training and evaluation, including logs and charts.
- **skin_samples/**: Directory to place new images for classification.
- **src/**: Contains all the scripts needed to run the project.
- **requirements.txt**: Lists all the dependencies needed.

---

## Dependencies

Ensure all dependencies are installed:

- **PyTorch**: For model building and training.
- **Torchvision**: Provides access to pre-trained models and image transformations.
- **Scikit-learn**: For evaluation metrics.
- **Pandas**: For data manipulation and logging.
- **Matplotlib**: For plotting loss curves.
- **OpenPyXL**: To save logs in Excel format.
- **Pillow**: For image processing.

Install all dependencies using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, create one with the following content:

```
torch
torchvision
scikit-learn
pandas
matplotlib
openpyxl
Pillow
```

---

## Potential Improvements

- **Data Expansion**: Acquire more images to enhance model training and performance.
- **Advanced Architectures**: Experiment with deeper networks like ResNet-34 or ResNet-50 for potentially better feature extraction.
- **Hyperparameter Tuning**: Implement techniques like grid search to find optimal hyperparameters.
- **Cross-Validation**: Use k-fold cross-validation to improve the robustness of the model.
- **Integration with Clinical Data**: Incorporate patient metadata to provide a more comprehensive diagnostic tool.
- **Web Application**: Develop a user-friendly web interface for easier accessibility.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:

1. **Fork the Repository**

   Click on the 'Fork' button on the top right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/QuickAid.git
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes and Commit**

   ```bash
   git add .
   git commit -m "Add your commit message"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request**

   Go to the original repository and create a pull request from your fork.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or inquiries, please contact:

- **Allen J Chandler**
  - Email: allenjchandler@vt.edu
           allen@shiftedorigin.com
           allenjchandler@gmail.com
  - GitHub: [wonmorewave](https://github.com/wonmorewave)

- **Jordan A Holmes**
  - Email: jah26603@vt.edu
  - GitHub: [dogblazer](https://github.com/dogblazer)
---

**Disclaimer**: This project is intended for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.
