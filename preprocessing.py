import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('C:\Users\aakri\Desktop\pcl 5th sem\preprocessing.py')

# Display the first few rows of the dataset
print(data.head())

# Preprocessing
# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop('Total Score', axis=1)  # Replace 'Total Score' with your target variable name if different
y = data['Total Score']  # Assuming 'Total Score' is your target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)