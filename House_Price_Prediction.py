# Importing the necessary libraries
import numpy as np  # For array operations
import pandas as pd  # For working with dataframes
import matplotlib.pyplot as plt  # For creating plots and graphs
import seaborn as sns  # For data visualization
import sklearn.datasets  # Provides datasets and machine learning algorithms
from sklearn.model_selection import train_test_split  # Splits data into train and test sets
from xgboost import XGBRegressor  # XGBoost Regressor for regression
from sklearn import metrics  # For evaluating the model

from sklearn.datasets import fetch_california_housing  # Fetches California housing dataset

# Fetching the California housing dataset
california_housing = fetch_california_housing()
print(california_housing)

# Loading the dataset into a Pandas DataFrame
california_housing_dataframe = pd.DataFrame(california_housing.data)
print(california_housing_dataframe)  # Displaying DataFrame without feature names

# Displaying the first 5 rows of the DataFrame
california_housing_dataframe.head()

# Adding feature names to the DataFrame
california_housing_dataframe = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_housing_dataframe.head()

# Adding the target (price) column to the DataFrame
california_housing_dataframe["price"] = california_housing.target
california_housing_dataframe.head()

# Checking the number of rows and columns in the data frame
california_housing_dataframe.shape

# Checking for missing values in the DataFrame
california_housing_dataframe.isnull().sum()

# Computing statistical measures of the dataset (mean, median, mode, percentiles, range, variance, and standard deviation)
california_housing_dataframe.describe()

# Understanding the correlation between various features in the dataset
correlation = california_housing_dataframe.corr()

# Constructing a heatmap to visualize the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size': 8}, cmap='Blues')

# Splitting the data and target
x = california_housing_dataframe.drop(['price'], axis=1)
y = california_housing_dataframe['price']

print(x)
print(y)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(x.shape, X_train.shape, X_test.shape)

# Model Training using XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, Y_train)

# Evaluating the model on the training data
training_data_prediction = model.predict(X_train)

print(training_data_prediction)
print(X_train)

# Computing R-squared error and Mean Absolute Error for training data
score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error ", score_1)
print("Mean Absolute error ", score_2)

# Visualizing the actual Prices vs Predicted Prices for the training data
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Prediction on Test Data

# Evaluating the model on the test data
test_data_prediction = model.predict(X_test)

# Computing R-squared error and Mean Absolute Error for test data
score_1 = metrics.r2_score(Y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error ", score_1)
print("Mean Absolute error ", score_2)

# Visualizing the actual Prices vs Predicted Prices for the test data
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()
