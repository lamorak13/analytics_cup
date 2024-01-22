import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


# Load data
diet_df = pd.read_csv('diet.csv')
recipes_df = pd.read_csv('recipes.csv')
reviews_df = pd.read_csv('reviews.csv', low_memory=False)
requests_df = pd.read_csv('requests.csv')

print("________________________________")
print("DATA EXPLORATION OF EACH DATASET")
print("________________________________")
print("")

# Explore each dataset
print("Diet DataFrame:")
print("")
print(diet_df.head())
print("______________________________")
print("sum of null values:")
print(diet_df.isnull().sum())
print("______________________________")
print(diet_df.describe())

print("________________________________________________________________________________")

print("\nRecipes DataFrame:")
print("")
print(recipes_df.head())
print("______________________________")
print("sum of null values:")
print(recipes_df.isnull().sum())
print("______________________________")
print(recipes_df.describe())

print("________________________________________________________________________________")

print("\nReviews DataFrame:")
print("")
print(reviews_df.head())
print("______________________________")
print("sum of null values:")
print(reviews_df.isnull().sum())
print("______________________________")
print(reviews_df.describe())

print("________________________________________________________________________________")

print("\nRequests DataFrame:")
print("")
print(requests_df.head())
print("______________________________")
print("sum of null values:")
print(requests_df.isnull().sum())
print("______________________________")
print(requests_df.describe())

print("________________________________________________________________________________")


# Merging reviews with diet on AuthorId
merged_df = pd.merge(reviews_df, diet_df, on='AuthorId', how='left')

# Merging with recipes on RecipeId
merged_df = pd.merge(merged_df, recipes_df, on='RecipeId', how='left')

# Merging with requests on both AuthorId and RecipeId
merged_df = pd.merge(merged_df, requests_df, on=['AuthorId', 'RecipeId'], how='left')

print("\nMerged DataFrame:")
print("")
print(merged_df.head())
print("______________________________")
print("sum of null values:")
print(merged_df.isnull().sum())
print("______________________________")
print(merged_df.describe())

# Handling missing values
#merged_df.fillna(merged_df.mean(), inplace=True)  # Numerical columns
#merged_df.fillna('Unknown', inplace=True)  # Categorical columns

# Convert categorical variables to numeric
#merged_df = pd.get_dummies(merged_df, columns=['Diet', 'RecipeCategory'])

print("________________________________________________________________________________")



# Feature Engineering
# 1. Total Time = CookTime + PrepTime
merged_df['TotalTime'] = merged_df['CookTime'] + merged_df['PrepTime']

# 2. Number of Ingredients (example, adjust based on actual data structure)
merged_df['NumIngredients'] = merged_df['RecipeIngredientParts'].apply(lambda x: len(str(x).split(',')))

# 3. Categorizing Nutritional Content
# Example: High Calorie if calories > 500
merged_df['HighCalorie'] = merged_df['Calories'].apply(lambda x: 1 if x > 500 else 0)

# 4. Creating Age Groups
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
merged_df['AgeGroup'] = pd.cut(merged_df['Age'], bins=bins, labels=labels)

# 5. Interaction Feature (Example)
merged_df['Diet_RecipeCategory'] = merged_df['Diet'] + "_" + merged_df['RecipeCategory']

# merged_df.drop(['CookTime', 'PrepTime', 'RecipeIngredientParts', 'Age'], axis=1, inplace=True)

print("\nMerged and engineered DataFrame:")
print("")
print(merged_df.describe())


""" ANDERER ANSATZ / RUMSPIELEREI AM ANFANG (EHER TRASH) """

# Load data
reviews = pd.read_csv('reviews.csv', low_memory=False)

# Drop rows with NaN values in either 'Rating' or 'Like' columns
reviews.dropna(subset=['Rating', 'Like'], inplace=True)

# Define X and y
X = reviews[['Rating']]  # Features
y = reviews['Like']      # Target

# split the data
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


