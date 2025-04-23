"""Perform the following operations using Python on any open source dataset (e.g., data.csv)
1. Import all the required Python Libraries.
2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear
description of the data and its source (i.e., URL of the web site).
3. Load the Dataset into pandas dataframe.
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
function to get some initial statistics. Provide variable descriptions. Types of variables etc.
Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking the
data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set.
If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables in Python.
In addition to the codes and outputs, explain every operation that you do in the above steps and explain
everything that you do to import/read/scrape the data set."""


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\userp\Downloads\archive (4)\iris.csv')

print("First 5 rows of the dataset:")
print(df.head())

print("\nColumn names in the dataset:")
print(df.columns)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDescriptive statistics for the dataset:")
print(df.describe())

print("\nDataset dimensions (rows, columns):", df.shape)

print("\nData types of each column:")
print(df.dtypes)

df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

df['sepal_length'].fillna(df['sepal_length'].mean(), inplace=True)
df['sepal_width'].fillna(df['sepal_width'].mean(), inplace=True)
df['petal_length'].fillna(df['petal_length'].mean(), inplace=True)
df['petal_width'].fillna(df['petal_width'].mean(), inplace=True)

print("\nAfter converting 'species' to numeric:")
print(df[['species']].head())