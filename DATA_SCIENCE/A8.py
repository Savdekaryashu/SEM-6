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
# Plot the correlation matrix to see relationships between numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Titanic Dataset')
plt.show()

# Explore 'fare' against different categorical variables (e.g., 'class', 'embarked', etc.)
sns.boxplot(x='class', y='fare', data=titanic)
plt.title('Fare Distribution by Class')
plt.show()