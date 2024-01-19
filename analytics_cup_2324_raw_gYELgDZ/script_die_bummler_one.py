import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Load the data
reviews = pd.read_csv('reviews.csv', low_memory=False)

X = reviews[['Rating']]  # features
y = reviews['Like']          # target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
print(f"Linear Regression MSE: {linear_mse}")


# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_report = classification_report(y_test, logistic_predictions)
print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print(logistic_report)
