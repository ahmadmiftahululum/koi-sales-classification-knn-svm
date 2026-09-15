# 🐟 Koi Fish Sales Classification using KNN and SVM

This repository contains a Machine Learning project that classifies and predicts **Koi Fish Prices** based on various physical features and specifications. The project benchmarks two classical classification models: **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Model Architecture & Algorithms](#-model-architecture--algorithms)
- [Evaluation Metrics](#-evaluation-metrics)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage](#-usage)

---

## 🧮 Overview
The goal of this project is to categorize/predict the market price (`Harga`) of Koi fish using feature representations such as variety, size, gender, age, and specification origin (Local vs Import).

---

## 📁 Dataset
* **Source:** Custom internal dataset (`KOI2.csv`).
* **Features:**
  * `Jenis Ikan`: Koi varieties (Goromo, Kohaku, Sanke, Tancho, Shiro, Showa, Utsuri, Shushui, Chagoi, Platinum)
  * `Ukuran`: Length / size of the Koi fish
  * `Gender`: Male or Female
  * `Umur`: Age group (`< 1 year`, `1 - 3 years`, `> 3 years`)
  * `Spesifikasi`: Origin (`Local`, `Import`)
* **Target Variable:** `Harga` (Koi Price)

---

## 🧹 Data Preprocessing
1. **Feature Selection:** Dropped irrelevant or highly missing columns (`Jumlah Corak`, `Corak Dominan`, `Tempat`).
2. **Label Encoding & Mapping:** Mapped categorical string features (Variety, Gender, Age, Specification) into numerical variables.
3. **Missing Value Analysis:** Verified zero null/missing values across features.
4. **Data Splitting:** Divided dataset into **80% Training** and **20% Testing** sets (`random_state=10`).

---

## 🤖 Model Architecture & Algorithms

### 1. K-Nearest Neighbors (KNN)
* **K-Neighbors:** $k = 3$
* Predicts class labels based on majority vote of the 3 nearest data points in feature space.

### 2. Support Vector Machine (SVM)
* **Kernel:** Linear (`kernel='linear'`)
* Constructs an optimal hyperplane to separate target price classes.

---

## 📊 Evaluation Metrics
Both models are evaluated on the test set using:
* **Accuracy Score:** Percentage of correct predictions.
* **Confusion Matrix:** Detail view of true positives/negatives across classes.
* **Classification Report:** Precision, Recall, and F1-Score breakdown.

---

## 🛠️ Prerequisites & Installation

Ensure Python 3.8+ is installed along with the required dependencies:

```bash
pip install pandas numpy scikit-learn
