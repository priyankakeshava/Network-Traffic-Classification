# Network Traffic Classification

Multi-class network traffic classification (BENIGN, LDAP, and NetBIOS) using Python, Scikit-learn, and Random Forest.

## Overview

This project applies supervised machine learning techniques to classify network traffic into three categories:

* **BENIGN** – Normal network traffic
* **LDAP** – LDAP-based traffic
* **NetBIOS** – NetBIOS-based traffic

The project uses a cleaned subset of the CIC-DDoS2019 dataset and focuses on understanding traffic behavior through feature engineering, visualization, and machine learning classification.

---

## Project Workflow

### 1. Data Cleaning and Preprocessing

* Removed unnecessary features from the original dataset.
* Handled missing values.
* Selected relevant network traffic features.
* Extracted and balanced samples from BENIGN, LDAP, and NetBIOS classes.

### 2. Feature Analysis

* Feature selection from over 80 original network traffic features.
* Correlation analysis to identify redundant features.
* Visualization of feature relationships and distributions.

### 3. Model Development

* Label encoding for multi-class classification.
* Train-test split for model evaluation.
* Random Forest classifier implementation.
* Model evaluation using multiple performance metrics.

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Results

The Random Forest classifier achieved approximately **99.97% accuracy** on the processed dataset.

### Key Findings

* Packet-length-related features contributed most to classification performance.
* Strong separation exists between BENIGN, LDAP, and NetBIOS traffic patterns.
* Several packet-related features were highly correlated and provided similar information.
* Feature importance analysis showed that packet size and packet length features dominated the model's decision-making process.

---

## Visualizations

The project includes the following visual analyses:

* Class Distribution
* Confusion Matrix
* Feature Importance
* Packet Length Mean vs Label (Box Plot)
* Correlation Heatmap

Generated graphs can be found in the **Visuals/** directory.

---

## Repository Structure

```text
Network-Traffic-Classification/
│
├── Cleaning.py          # Dataset cleaning and preprocessing
├── model.py             # Model training and evaluation
├── Visuals/             # Generated graphs and visualizations
├── README.md
├── LICENSE
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## Dataset

The dataset used for this project is derived from the CIC-DDoS2019 dataset.

The cleaned dataset used during experimentation is not included in this repository.

---

## Conclusion

The results demonstrate that Random Forest can effectively classify BENIGN, LDAP, and NetBIOS traffic with very high accuracy. Feature analysis indicates that packet-size-related characteristics play a significant role in distinguishing different traffic types, while several highly correlated features contribute redundant information.
