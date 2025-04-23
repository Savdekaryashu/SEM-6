"""1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about
the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we
can find any patterns in the data.
2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
is distributed by plotting a histogram"""
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')

# Plot the distribution of 'fare' to check how the ticket prices are distributed
plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], kde=True, bins=30, color='blue')
plt.title('Distribution of Ticket Fare on Titanic')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Optional
# Additional exploration to find patterns in the data
# Plot the correlation matrix with only numeric columns
plt.figure(figsize=(10, 8))
numeric_data = titanic.select_dtypes(include='number')  # Select only numeric columns
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Columns in Titanic Dataset')
plt.show()

# Explore 'fare' against different categorical variables (e.g., 'class', 'embarked', etc.)
sns.boxplot(x='class', y='fare', data=titanic)
plt.title('Fare Distribution by Class')
plt.show()
