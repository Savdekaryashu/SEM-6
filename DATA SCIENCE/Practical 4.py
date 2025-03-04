import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
train_df = pd.read_csv("D:/STUDY/DATA SCIENCE/Data/Boston/train.csv")
test_df = pd.read_csv("D:/STUDY/DATA SCIENCE/Data/Boston/test.csv")

# Drop ID column as it's not useful for training
test_ids = test_df['ID']  # Save test IDs for output
test_df = test_df.drop(columns=['ID'])
train_df = train_df.drop(columns=['ID'])

# Define features and target variable
X = train_df.drop(columns=['medv'])  # Features
y = train_df['medv']  # Target variable

# Split data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = model.predict(X_val)

# Evaluate model performance
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Predict on test set
test_predictions = model.predict(test_df)

# Save predictions to CSV
output = pd.DataFrame({'ID': test_ids, 'Predicted_MEDV': test_predictions})
output.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")

# Visualizing Actual vs Predicted Prices (on validation data)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_val, y=y_val_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Validation Data)")
plt.show()
