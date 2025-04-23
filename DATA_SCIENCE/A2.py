"""Create an “Academic performance” dataset of students and perform the following operations using
Python.
1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
inconsistencies, use any of the suitable techniques to deal with them.
2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques
to deal with them.
3. Apply data transformations on at least one of the variables. The purpose of this
transformation should be one of the following reasons: to change the scale for better
understanding of the variable, to convert a non-linear relation into a linear one, or to decrease
the skewness and convert the distribution into a normal distribution.
Reason and document your approach properly"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\userp\Downloads\archive (10)\StudentsPerformance.csv")

# Step 1: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Show initial data
print("Initial Dataset Info:")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

df.loc[2, 'math_score'] = np.nan
df['math_score'].fillna(df['math_score'].mean(), inplace=True)

# Step 2: Detect and handle outliers using Z-score for numeric columns
numeric_cols = ['math_score', 'reading_score', 'writing_score']
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]

# Step 3: Data transformation (log transformation on reading_score)
df_no_outliers['log_reading_score'] = np.log1p(df_no_outliers['reading_score'])

# Show skewness before and after
print("\nSkewness Before:", df_no_outliers['reading_score'].skew())
print("Skewness After:", df_no_outliers['log_reading_score'].skew())

# Final cleaned and transformed data
print("\nFinal Cleaned Dataset Sample:\n", df_no_outliers.head())
