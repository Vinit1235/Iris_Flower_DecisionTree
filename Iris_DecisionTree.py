"""
Iris Dataset Classification using Decision Tree
This script trains a Decision Tree classifier on the Iris dataset
and visualizes the resulting decision tree.
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create DataFrame for better data exploration
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display dataset overview
print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nSpecies distribution:\n{df['species'].value_counts()}")

# Prepare features (X) and target (Y)
X = iris.data  # Using original data directly
Y = iris.target

# Split data into training and testing sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# Make predictions on test set
Y_pred = model.predict(X_test)

# Calculate and display accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(model, 
               feature_names=iris.feature_names, 
               class_names=iris.target_names, 
               filled=True,
               rounded=True)
plt.title("Decision Tree Classifier - Iris Dataset", fontsize=16, pad=20)
plt.tight_layout()
plt.show()


