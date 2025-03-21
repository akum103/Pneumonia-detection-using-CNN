🩺 Pneumonia Detection using CNN

📌 Overview

Pneumonia is a leading cause of mortality worldwide, particularly among children and the elderly. Early detection plays a crucial role in effective treatment. This project implements a Convolutional Neural Network (CNN) model to automate pneumonia detection from chest X-ray images, reducing diagnostic time and improving accuracy. The study compares a custom CNN model with a pre-trained ResNet50V2 model.

🎯 Business Objective

Automate pneumonia diagnosis using deep learning on chest X-ray images.

Reduce turnaround time for radiology reports.

Improve accuracy and recall compared to traditional diagnosis.

📂 Dataset

Source: Kaggle - Chest X-ray Images (Pneumonia)

Size: 5,856 X-ray images

Classes:

Normal (No Pneumonia)

Pneumonia (Bacterial/Viral Infections)

Splits:

Train: 5,216 images

Validation: 16 images

Test: 624 images

🏗️ Model Architectures

This study evaluates two deep learning models:

1️⃣ Pre-trained ResNet50V2 Model

ResNet (Residual Networks) uses skip connections to improve gradient flow.

Fine-tuned on pneumonia dataset.

Image resolution: 128×128 pixels

Performance:

Accuracy: 56.2%

Recall: 77%

2️⃣ Custom CNN Model (Proposed Approach)

Built from scratch to learn pneumonia-specific features.

5 Convolutional layers with ReLU activation.

Max pooling for dimensionality reduction.

Dropout (20%) to prevent overfitting.

Image resolution: 224×224 pixels

Performance:

Accuracy: 79.6%

Recall: 82% ✅ (Better than ResNet)

🚀 Workflow & Methodology

1️⃣ Data Preprocessing

Resized images to 224×224 pixels for CNN and 128×128 pixels for ResNet.

Data Augmentation (Rotation, Zoom, Flipping) to prevent overfitting.

Converted images to grayscale for reduced dimensionality.

2️⃣ Training & Validation

Train-Test Split: 80% training, 20% testing.

Loss Function: Binary Cross Entropy.

Optimizer: Adam Optimizer (Learning Rate = 0.001).

Batch Size: 32.

Epochs: 5 (Optimal epoch based on training results).

3️⃣ Performance Evaluation

Model

Accuracy

Precision

Recall

F1 Score

ResNet50V2

56.2%

63%

77%

69%

Custom CNN

79.6%

63%

82%

72%

4️⃣ Confusion Matrix Analysis

True Positives (TP): 321 pneumonia cases correctly identified.

False Negatives (FN): 69 cases misclassified as normal.

Class Imbalance Observed:

Train Set: 74% Pneumonia, 26% Normal.

Test Set: 62% Pneumonia, 38% Normal.

Validation Set: 50/50 split.

📊 Key Findings

✅ Custom CNN model outperforms ResNet50V2 in recall (82% vs 77%).
✅ Higher parameter count in CNN leads to better feature extraction.
✅ Addressing class imbalance could further improve accuracy.
✅ Automated pneumonia detection can significantly reduce diagnosis time.

🛠️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/akum103/pneumonia-detection.git
cd pneumonia-detection

2️⃣ Install Dependencies

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn

3️⃣ Run the Model

jupyter notebook CNN_Model.ipynb

📄 Project Report

📥 Download the full report

🚀 Future Improvements

Enhance training dataset by adding more X-ray images.

Implement Transfer Learning using EfficientNet for better feature extraction.

Deploy as a Web App using Flask for real-time diagnosis.

Apply Generative Adversarial Networks (GANs) to generate synthetic X-rays.

💡 Author: akum103🎯 GitHub Repo: pneumonia-detection
