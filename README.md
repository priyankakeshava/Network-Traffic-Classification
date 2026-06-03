# Network-Traffic-Classification
Multi-class network traffic classification (BENIGN, LDAP, NetBIOS) using Python, Scikit-learn, and Random Forest.
# Machine Learning-Based Network Intrusion Detection

This project focuses on classifying network traffic into BENIGN, LDAP, and NetBIOS categories using supervised machine learning techniques.

The analysis was performed on the CIC-DDoS2019 dataset and includes:

- Data cleaning and preprocessing
- Feature selection from 80+ network traffic features
- Class balancing to reduce dataset bias
- Exploratory Data Analysis (EDA)
- Random Forest model training and evaluation
- Feature importance analysis
- Correlation analysis
- Performance evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

The model achieved approximately 99.97% classification accuracy, with packet-size-related features emerging as the most significant predictors of network traffic behavior.

## Key Findings

- Packet-length-related features contributed most to classification performance.
- The dataset exhibited significant class imbalance.
- Strong feature separation between BENIGN, LDAP, and NetBIOS traffic enabled high model accuracy.
- Several highly correlated features provided redundant information.
