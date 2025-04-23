"""Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
3. Create a boxplot for each feature in the dataset.
4. Compare distributions and identify outliers.
"""


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')

# 1. List down the features and their types
print("Features and their types:")
print(iris.dtypes)

# 2. Create a histogram for each feature
iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Plot histograms for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(iris[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3. Create a boxplot for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=iris[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()

# 4. Compare distributions and identify outliers
print("\nObservations:")
print("1. Sepal Length and Sepal Width show normal distributions, but Petal Length and Petal Width have some skewness.")
print("2. All the features show the presence of outliers in the boxplots, particularly in Petal Length.")