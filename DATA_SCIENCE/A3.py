"""Perform the following operations on any open source dataset (e.g., data.csv)
1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a
Curriculum for Third Year of Artificial Intelligence and Data Science (2019 Course), Savitribai Phule Pune University
http://collegecirculars.unipune.ac.in/sites/documents/Syllabus2022/Forms/AllItems.aspx #84/105
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
# Example: gender -> numeric mapping
gender_map = {'female': 0, 'male': 1}
gender_numeric = df['gender'].map(gender_map).tolist()
print("\nMapped Gender to Numeric Values:\n", gender_numeric[:10])  # Display first 10

import seaborn as sns

# Load iris dataset
iris = sns.load_dataset('iris')

# Display basic statistics for each species
species_stats = iris.groupby('species').describe(percentiles=[.25, .5, .75])
print("\nBasic Statistical Details of Each Iris Species:\n", species_stats)