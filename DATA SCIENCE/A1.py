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