import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Load data
diet_df = pd.read_csv('diet.csv')
recipes_df = pd.read_csv('recipes.csv')
reviews_df = pd.read_csv('reviews.csv', low_memory=False)
requests_df = pd.read_csv('requests.csv')

'''

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

'''

print("________________________________")
print("DATA PREPARATION")
print("________________________________")
print("")

# Merging reviews with diet on AuthorId
merged_df = pd.merge(reviews_df, diet_df, on='AuthorId', how='left')

# Merging with recipes on RecipeId
merged_df = pd.merge(merged_df, recipes_df, on='RecipeId', how='left')

# Merging with requests on both AuthorId and RecipeId
merged_df = pd.merge(merged_df, requests_df, on=['AuthorId', 'RecipeId'], how='left')

'''
print("\nMerged DataFrame:")
print("")
print(merged_df.head())
print("______________________________")
print("sum of null values:")
print(merged_df.isnull().sum())
print("______________________________")
print(merged_df.describe())
'''

print("________________________________________________________________________________")



''' Feature Engineering / Data preparation '''

# Total Time = CookTime + PrepTime
merged_df['TotalTime'] = merged_df['CookTime'] + merged_df['PrepTime']


# Number of Ingredients (example, adjust based on actual data structure)
merged_df['NumIngredients'] = merged_df['RecipeIngredientParts'].apply(lambda x: len(str(x).split(',')))


# Categorizing Nutritional Content
# Example: High Calorie if calories > 500
merged_df['HighCalorie'] = merged_df['Calories'].apply(lambda x: 1 if x > 500 else 0)


# Creating Age Groups
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
merged_df['AgeGroup'] = pd.cut(merged_df['Age'], bins=bins, labels=labels)


# fill NaN value in diet of the one entry that has no diet
if merged_df['Diet'].isna().any():
    merged_df['Diet'].fillna("Vegetarian", inplace=True)


# fill NaN values in RecipeServings with the median of servings
# TODO: decide if to fill with median or mean or drop?
servings_median = merged_df['RecipeServings'].median()
servings_mean = merged_df['RecipeServings'].mean()
merged_df["RecipeServings"].fillna(servings_median, inplace=True)
# merged_df["RecipeServings"].dropna()              # can be activated if needed


# drop the Rating variable from the merged as well as reviews dataframe (that will be split soon)
# as it has no meaningful values (just 2.0 or NaN) and therefore serves no purpose
rating_column = 'Rating'
if rating_column in merged_df.columns and rating_column in reviews_df.columns:
    merged_df.drop(rating_column, axis=1, inplace=True)
    reviews_df.drop(rating_column, axis=1, inplace=True)
else:
    print("There has been an issue with the columns, please check!")

# Possibly drop all variables that are not needed for further modeling
# merged_df.drop(['CookTime', 'PrepTime', 'RecipeIngredientParts', 'Age'], axis=1, inplace=True)

print("\nMerged and engineered DataFrame:")
print("")
print(merged_df.describe())
print(merged_df.columns)


''' Splitting of the reviews.csv file into prediction, training and testing sets '''

split_index_training = 42814
split_index_testing = 42814 + int((140195 - 42814) / 2)

if not (0 <= split_index_training < split_index_testing <= len(reviews_df)):
    raise ValueError("Out of bounds or wrong order.")

prediction_df = reviews_df.iloc[:split_index_training]
training_df = reviews_df.iloc[split_index_training:split_index_testing]
testing_df = reviews_df.iloc[split_index_testing:]

print("\nPrediction DataFrame:")
print("")
print(prediction_df.head())
print(prediction_df.describe())
print(prediction_df.columns)

print("\nTraining DataFrame:")
print("")
print(training_df.head())
print(training_df.describe())
print(training_df.columns)

print("\nTesting DataFrame:")
print("")
print(testing_df.head())
print(testing_df.describe())
print(testing_df.columns)
print("")
print("")


print("________________________________________________________________________________")


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
#logistic_model.fit(X_train, y_train)

#logistic_predictions = logistic_model.predict(X_test)

#logistic_accuracy = accuracy_score(y_test, logistic_predictions)
#logistic_report = classification_report(y_test, logistic_predictions)
#print(f"Logistic Regression Accuracy: {logistic_accuracy}")
#print(logistic_report)