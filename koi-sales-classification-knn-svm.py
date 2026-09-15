# -*- coding: utf-8 -*-
"""
Koi Fish Sales Classification using KNN and SVM
------------------------------------------------
This script processes Koi fish dataset features (variety, size, age, origin)
and classifies target prices using KNN and SVM algorithms.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. DATA GATHERING & LOADING
# ==========================================
# Define dataset path (Local directory or Google Drive)
DATA_PATH = 'KOI2.csv'

if os.path.exists(DATA_PATH):
    koi = pd.read_csv(DATA_PATH, delimiter=';')
    print("Dataset successfully loaded from local path.")
else:
    # Google Colab Drive Mount Fallback
    from google.colab import drive
    drive.mount('/content/drive')
    koi = pd.read_csv('drive/MyDrive/Tugas Machine Learning/Tugas_UAS/KOI2.csv', delimiter=';')
    print("Dataset successfully loaded from Google Drive.")

# Display initial dataset structure
print("\n--- Initial Dataset Preview ---")
print(koi.head())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS & PREPROCESSING
# ==========================================
# Drop unnecessary columns
columns_to_drop = ['Jumlah Corak', 'Corak Dominan', 'Tempat']
koi.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

print("\n--- Dataset After Dropping Unnecessary Columns ---")
print(koi.head())

# Display descriptive statistics for numerical attributes
print("\n--- Summary Statistics ---")
print(koi.describe())

# Categorical Feature Mapping to Numerical Values
koi['Jenis Ikan'] = koi['Jenis Ikan'].map({
    'Goromo': 0, 'Kohaku': 1, 'Sanke': 2, 'Tancho': 3, 'Shiro': 4,
    'Showa': 5, 'Utsuri': 6, 'Shushui': 7, 'Chagoi': 8, 'Platinum': 9
})
koi['Gender'] = koi['Gender'].map({'Female': 0, 'Male': 1})
koi['Umur'] = koi['Umur'].map({
    'Dibawah 1 tahun': 0, '1 - 3 Tahun': 1, 'Diatas 3 Tahun': 2
})
koi['Spesifikasi'] = koi['Spesifikasi'].map({'Local': 0, 'Import': 1})

# Apply Label Encoder for numerical transformation
encoder = LabelEncoder()
for col in ['Jenis Ikan', 'Gender', 'Umur', 'Spesifikasi']:
    if col in koi.columns:
        koi[col] = encoder.fit_transform(koi[col])

print("\n--- Encoded Dataset Preview ---")
print(koi.head())

# Check for missing values
missing_count = koi.isnull().sum().sum()
print(f"\nTotal Missing Values in Dataset: {missing_count}")

# Define Features (X) and Target Label (y)
X = koi[['Jenis Ikan', 'Ukuran', 'Gender', 'Umur', 'Spesifikasi']]
y = koi['Harga']

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")

# ==========================================
# 3. TRAIN / TEST DATA SPLITTING
# ==========================================
# Split dataset into 80% Training and 20% Testing sets
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=10
)

print(f"\nTraining Set Shape: {train_X.shape}")
print(f"Testing Set Shape:  {test_X.shape}")

# ==========================================
# 4. K-NEAREST NEIGHBORS (KNN) CLASSIFICATION
# ==========================================
print("\n==========================================")
print("     K-NEAREST NEIGHBORS (KNN)")
print("==========================================")

# Instantiate KNN classifier (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X, train_y)

# Predict test set labels
pred_y_knn = knn.predict(test_X)

# Evaluate KNN Performance
accuracy_knn = accuracy_score(test_y, pred_y_knn)
print(f"KNN Model Accuracy: {accuracy_knn * 100:.2f}%\n")

print("KNN Confusion Matrix:")
print(confusion_matrix(test_y, pred_y_knn))

# ==========================================
# 5. SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION
# ==========================================
print("\n==========================================")
print("     SUPPORT VECTOR MACHINE (SVM)")
print("==========================================")

# Instantiate Linear SVM Classifier
svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(train_X, train_y)

# Predict test set labels
pred_y_svm = svm_model.predict(test_X)

# Evaluate SVM Performance
accuracy_svm = accuracy_score(test_y, pred_y_svm)
print(f"SVM Model Accuracy: {accuracy_svm * 100:.2f}%\n")

print("SVM Confusion Matrix:")
print(confusion_matrix(test_y, pred_y_svm))

print("\nSVM Detailed Classification Report:")
print(classification_report(test_y, pred_y_svm, zero_division=0))