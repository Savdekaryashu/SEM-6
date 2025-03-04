import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset (ensure 'titanic.csv' is in your working directory)
df = pd.read_csv("D:/STUDY/DATA SCIENCE/Data/test.csv")

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Display basic descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Print dataset dimensions and data types
print("\nDataset Dimensions (Rows, Columns):", df.shape)
print("\nData Types:")
print(df.dtypes)

# Convert 'PassengerId' to string if necessary (as it's an identifier)
df['PassengerId'] = df['PassengerId'].astype(str)

# Convert categorical variables into numerical values using LabelEncoder
le = LabelEncoder()

# Encode 'Sex'
df['Sex_encoded'] = le.fit_transform(df['Sex'])
print("\nEncoded 'Sex' Column:")
print(df[['Sex', 'Sex_encoded']].drop_duplicates())

# Encode 'Embarked' (first, fill missing values with a placeholder)
df['Embarked'] = df['Embarked'].fillna('Unknown')
df['Embarked_encoded'] = le.fit_transform(df['Embarked'])
print("\nEncoded 'Embarked' Column:")
print(df[['Embarked', 'Embarked_encoded']].drop_duplicates())

# Final DataFrame preview
print("\nFinal Processed DataFrame Preview:")
print(df.head())
