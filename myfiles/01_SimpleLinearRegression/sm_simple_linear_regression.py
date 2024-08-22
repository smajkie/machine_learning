# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#dataset = pd.read_csv('Salary_Data.csv')        # only numerical values there 
dataset = pd.read_csv('../../datasets/employee_data.csv')      # we need to transform categorical data here

# Depending on dataset maybe we need to get rid of categorical data

dataset = pd.get_dummies(dataset, columns=['Gender', 'Position'], drop_first=True)

# Printing the dataset after encoding
print("Dataset after encoding:\n", dataset.iloc[:, 2])


y = dataset.iloc[:, 2].values          # to jest nasz wzor jako target do predykcji
X = dataset.drop(columns=dataset.columns[2]).values         # to są nasze features

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Printing the predicted values next to the real ones
print("Real vs Predicted values:")
for real, predicted in zip(y_test, y_pred):
    print(f"Real: {real:.2f}, Predicted: {predicted:.2f}")

# Plotting the Real vs Predicted values
plt.scatter(y_test, y_pred, color='red')
plt.plot(y_test, y_test, color='blue')  # Line for perfect prediction
plt.title('Real vs Predicted values')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()