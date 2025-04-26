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
import matplotlib.pyplot as plt
import seaborn as sns

# Create Academic Performance dataset
data = {
    'Student_ID': [101, 102, 103, 104, 105],
    'Math_Score': [88, 92, np.nan, 70, 45],
    'Science_Score': [91, 85, 79, np.nan, 48],
    'English_Score': [86, 90, 78, 80, 999],  # 999 is an inconsistency (outlier)
    'Attendance_Rate': [0.95, 0.85, 0.75, np.nan, 0.65]
}

df = pd.DataFrame(data)

# Show the dataset
print(df)

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing Math_Score, Science_Score, Attendance_Rate with mean
df.fillna({"Math_Score":df['Math_Score'].mean()}, inplace=True)
df.fillna({'Science_Score':df['Science_Score'].mean()}, inplace=True)
df.fillna({'Attendance_Rate':df['Attendance_Rate'].mean()}, inplace=True)

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for outliers in English_Score
Q1 = df['English_Score'].quantile(0.25)
Q3 = df['English_Score'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(f"\nLower limit: {lower_limit}, Upper limit: {upper_limit}")

# Identify outliers
outliers = df[(df['English_Score'] < lower_limit) | (df['English_Score'] > upper_limit)]
print("\nOutliers:\n", outliers)

# Boxplot for English_Score
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['English_Score'])
plt.title('Boxplot of English Score (After Outlier Treatment)')
plt.xlabel('English Score')
plt.show()


# Replace 999 in English_Score
median_value = df['English_Score'].median()
df.loc[df['English_Score'] > upper_limit, 'English_Score'] = median_value

# Histogram for Math_Score (before transformation)
plt.figure(figsize=(8, 4))
sns.histplot(df['Math_Score'], kde=True, color='blue')
plt.title('Histogram of Math Score (Before Transformation)')
plt.xlabel('Math Score')
plt.show()

# Add a small value to avoid log(0)
df['Math_Score_Log'] = np.log(df['Math_Score'] + 1)

# Histogram for Math_Score_Log (after log transformation)
plt.figure(figsize=(8, 4))
sns.histplot(df['Math_Score_Log'], kde=True, color='green')
plt.title('Histogram of Math Score (After Log Transformation)')
plt.xlabel('Log of Math Score')
plt.show()

# Show the dataset after transformation
print("\nTransformed Dataset:\n", df)

