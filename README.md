# Network Traffic Classification

Multi-class network traffic classification (BENIGN, LDAP, and NetBIOS) using Python, Scikit-learn, and a Random Forest classifier.

## Overview

This project focuses on classifying network traffic into three categories:

* **BENIGN** – Normal network traffic
* **LDAP** – LDAP-based attack traffic
* **NetBIOS** – NetBIOS-based attack traffic

The analysis was performed using a subset of the **CIC-DDoS2019 dataset** and applies supervised machine learning techniques to identify and classify different traffic behaviors.

## Project Workflow

* Data cleaning and preprocessing
* Feature selection from 80+ network traffic features
* Class balancing to reduce dataset bias
* Exploratory Data Analysis (EDA)
* Random Forest model training
* Model evaluation and performance analysis
* Feature importance analysis
* Correlation analysis

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Results

The Random Forest classifier achieved approximately **99.97% accuracy** on the test dataset.

### Key Findings

* Packet-length-related features contributed the most to classification performance.
* The dataset initially exhibited significant class imbalance.
* Strong feature separation between BENIGN, LDAP, and NetBIOS traffic resulted in high classification accuracy.
* Several highly correlated features provided redundant information.
* Packet-size-related features were the most influential predictors of network traffic behavior.

## Repository Contents

* `ModelVisuals.py` – Model training, evaluation, and visualization generation
* `Classification of Network Traffic.pdf` – Detailed project report and analysis

## Dataset

The dataset used for this project is not included in this repository. The analysis was conducted on a cleaned subset of the CIC-DDoS2019 dataset containing BENIGN, LDAP, and NetBIOS traffic samples.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## Conclusion

The Random Forest model performed exceptionally well in distinguishing between BENIGN, LDAP, and NetBIOS traffic. The results indicate that packet-size-related features play a significant role in traffic classification, while several correlated features provide overlapping information that may be reduced in future work.
