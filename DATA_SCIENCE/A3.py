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