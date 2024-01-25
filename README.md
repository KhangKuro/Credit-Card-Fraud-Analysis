<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Cluster/blob/main/CCF.png" />

# Credit Card Fraud Analysis 💳🕵️‍♂️

## Introduction 🌟

Welcome to the introduction section of the credit card fraud behavior analysis study! In the rapidly evolving digital age, detecting and preventing fraud in financial transactions is becoming increasingly crucial. This study focuses on analyzing patterns and behaviors associated with credit card fraud to develop effective prevention strategies.

## Cluster Analysis Rationale 📊

The objective of this study is to understand the underlying patterns and characteristics of credit card fraud through cluster analysis. By grouping transactions into clusters based on various features such as transaction amount, time of day, and customer demographics, we aim to identify distinct fraud behavior profiles. This will enable financial institutions to better detect and mitigate fraudulent activities, ultimately enhancing security measures for credit card users.

<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Cluster/blob/main/Idea.png" />

## About the Dataset ℹ️

This dataset simulates credit card transactions, including both legitimate and fraudulent transactions, from January 1, 2019, to June 31, 2020. It includes information on transactions of 1000 customers using credit cards with a group of 800 service providers.

- [Kaggle - Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)

### Data Description 📝

- **trans_date_trans_time**: Date and time of the transaction.
- **cc_num**: Customer's credit card number.
- **merchant**: Service provider the customer is paying (e.g., Amazon, Walmart).
- **category**: Category of the transaction.
- **amt**: Amount of the transaction.
- **first, last**: Customer's first and last name.
- **gender**: Customer's gender.
- **street, city, state**: Customer's address.
- **zip**: Zip code of the transaction.
- **lat, long**: Customer's latitude and longitude.
- **city_pop**: Population of the city where the customer lives.
- **job**: Customer's occupation.
- **dob**: Customer's date of birth.
- **trans_num**: Unique transaction number for each transaction.
- **unix_time**: Transaction time in Unix format (usually not used in analysis due to uniqueness).
- **merch_lat, merch_long**: Latitude and longitude of the merchant.
- **is_fraud**: Binary index indicating whether the transaction is fraudulent (1 for fraud, 0 for not fraud).

<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Detection/blob/main/erd.png" />

## Import Libraries 📚

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from collections import Counter
from IPython.display import Image

# Import modules and classes from scikit-learn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings; warnings.filterwarnings("ignore")
```

## Import Data 📊

```python
# Read data from CSV files
fraud_train = pd.read_csv('/kaggle/input/fraud-detection/fraudTrain.csv')
fraud_test = pd.read_csv('/kaggle/input/fraud-detection/fraudTest.csv')

# Assign fraud_train to df, then drop the first column of fraud_test and df
df = fraud_train
fraud_test.drop(fraud_test.columns[:1], axis=1, inplace=True)
df.drop(df.columns[:1], axis=1, inplace=True)

# Display the first few rows of DataFrame df
df.head()
```
<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Cluster/blob/main/CCFCluster.png" />

## Insights 🧐

From the analysis of fraud groups, there are some important common points:

- **Transaction Time**: There is a concentration of transactions at specific times, including late night to early morning (midnight to 3 AM) and late evening (10-11 PM). This may reflect transaction activity during less monitored times.
  
- **Transaction Amount**: Groups differ in transaction values. Females perform significantly higher-value transactions - much more fraudulent than males, and most groups tend to focus on low-value transactions and large transactions for fraud.
  
- **Age**: Transaction-making ages are also diverse, but tend to be primarily from middle-aged to elderly (20 - 70 years old).
  
- **City Population**: There is diversity in the city population of the groups.

## Improvements for Fraud Prevention System 🛡️

1. **Enhanced Monitoring and Time Control**:
   - Focus on Special Time Frames: Enhance monitoring during specific time frames such as late night to early morning and late evening when fraudulent transactions may increase.
   - Stronger Time-Specific Monitoring: Strengthen controls on weekends and weekdays, when transaction volumes increase and fraud detection may be more challenging.

2. **Fraud Detection through Transaction Characteristics**:
   - Identify Unusual Transactions: Focus on transactions with large amounts during less monitored time frames to detect abnormal behavior.
   - Check Transactions by Gender: Identify differences in transaction values between males and females to determine fraudulent activities.

3. **Increase Knowledge and Tracking Systems**:
   - Enhance Information and Access Control: For occupations related to financial information, control access to personal and financial information of others.
   - Classify Transactions and Fraud Activities: Build machine learning models to classify common transaction types that may lead to fraud.

4. **Enhanced Monitoring of Location and Transaction Time**:
   - Focus on Monitoring Transaction Time and Location: Strengthen controls in cities with high population density and during periods of high transaction volume.
   - Monitor Time During the Year: Strengthen controls during holidays, busy shopping seasons, and peak transaction periods.

5. **Enhanced Fraud Prevention and Detection Technology**:
   - Apply Data Analysis Technology: Use advanced data analysis tools to detect and prevent fraud.
   - Expand Machine Learning Models: Build predictive machine learning models to classify and detect fraud based on data from previous fraudulent transactions.

## Conclusion 🎉

Through meticulous data analysis and understanding of fraudulent transaction characteristics, we are ready to build a predictive model. This model will be based on valuable information about how fraud occurs, thereby creating a more flexible and powerful tool to identify and prevent fraudulent behavior in the future.

By using carefully analyzed data and deep knowledge from the analysis process, we will proceed to build an accurate and efficient predictive model. This model will result from the combination of artificial intelligence and advanced machine learning methods, aiming to leverage and understand more about how fraud can exist in future transactions.

We will build a stronger, reliable, and more flexible predictive model to contribute to the development of a safer and more transparent digital transaction environment in the future.

For further details and code implementation, please refer to the respective Jupyter Notebook or Python script.


Sure, here's the introduction, cluster analysis rationale, acknowledgements, and disclaimer for the data science seminar prelude to the final thesis:

## Acknowledgements 🙏

We would like to express our sincere gratitude to the following sources for their valuable insights and resources:

- [Kaggle - Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Yellowbrick Documentation](https://www.scikit-yb.org/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

We are grateful for their contributions to the field of data science and machine learning, which have greatly enriched our research.

## Disclaimer ⚠️

This study is a part of a data science seminar and is intended for educational purposes only. The analysis and findings presented here are based on simulated data and should not be used for any commercial purposes or real-world decision-making without proper validation and authorization.

