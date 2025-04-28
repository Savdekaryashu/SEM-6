"""Perform the following operations on any open source dataset (e.g., data.csv)
1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a
dataset (age, income etc.) with numeric variables grouped by one of the qualitative
(categorical) variable. For example, if your categorical variable is age groups and quantitative
variable is income, then provide summary statistics of income grouped by the age groups.
Create a list that contains a numeric value for each response to the categorical variable.
2. Write a Python program to display some basic statistical details like percentile, mean,
standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of
iris.csv dataset.
Provide the codes with outputs and explain everything that you do in this step"""

import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\userp\Downloads\archive (10)\StudentsPerformance.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Summary statistics grouped by a categorical variable (e.g., gender)
grouped_stats = df.groupby('gender')[['math_score', 'reading_score', 'writing_score']].agg(['mean', 'median', 'min', 'max', 'std'])

print("Grouped Summary Statistics by Gender:\n", grouped_stats)

# Create a list that maps numeric values to categorical responses
# Create lists for age
male_age = list(df[df['gender'] == 'Male']['age'])
female_age = list(df[df['gender'] == 'Female']['age'])

print("Male Ages:", male_age[:10])
print("Female Ages:", female_age[:10])

#2) Iris 

# Step 1: Load the Iris dataset
df2 = pd.read_csv("G:/TE/DS/dtsets/IRIS.csv")  # <- change path if needed

# Step 2: Display first few rows
print(df2.head())

# Step 3: Filter species separately
setosa = df2[df2['species'] == 'Iris-setosa']
versicolor = df2[df2['species'] == 'Iris-versicolor']
virginica = df2[df2['species'] == 'Iris-virginica']

# Step 4: Display statistical details
print("\n--- Iris-setosa ---\n")
print(setosa.describe())

print("\n--- Iris-versicolor ---\n")
print(versicolor.describe())

print("\n--- Iris-virginica ---\n")
print(virginica.describe())
