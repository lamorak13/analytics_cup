import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Load data
diet_df = pd.read_csv('diet.csv')
recipes_df = pd.read_csv('recipes.csv')
reviews_df = pd.read_csv('reviews.csv', low_memory=False)
requests_df = pd.read_csv('requests.csv')

#merged_df = pd.merge(reviews_df, diet_df, on='AuthorId', how='left')
#merged_df = pd.merge(merged_df, recipes_df, on='RecipeId', how='left')
#merged_df = pd.merge(merged_df, requests_df, on=['AuthorId', 'RecipeId'], how='left')

print("________________________________________________________________________________")

''' Feature Engineering / Data preparation '''


def dropColumn(dataframe, column):
    if column in dataframe.columns:
        dataframe.drop(column, axis=1, inplace=True)
    else:
        print("There has been an issue with the columns in ", dataframe, ". Please check!")


'''DIET'''
# Fill NaN value in diet of the one entry that has no diet
if diet_df['Diet'].isna().any():
    diet_df['Diet'].fillna("Vegetarian", inplace=True)


'''RECIPES'''
# Total Time = CookTime + PrepTime
recipes_df['TotalTime'] = recipes_df['CookTime'] + recipes_df['PrepTime']
dropColumn(recipes_df, 'CookTime')
dropColumn(recipes_df, 'PrepTime')
dropColumn(recipes_df, 'RecipeYield')
dropColumn(reviews_df, 'Rating')
dropColumn(reviews_df, 'Like')

# Fill NaN values in RecipeServings with the median of servings
servings_median = recipes_df['RecipeServings'].median()
recipes_df["RecipeServings"].fillna(servings_median, inplace=True)


'''REQUESTS'''
# Replace all negative time values with absolute value
requests_df['Time'] = requests_df['Time'].apply(lambda x: abs(x) if x < 0 else x)

# Cast floats to int
requests_df['HighCalories'] = requests_df['HighCalories'].astype(int)

# Correct HighProtein column
requests_df['HighProtein'] = requests_df['HighProtein'].apply(
    lambda x: 1 if x == 'Yes' else (0 if x == 'Indifferent' else x))

# Correct LowSugar column
requests_df['LowSugar'] = requests_df['LowSugar'].apply(lambda x: 1 if x == 'Indifferent' else x)

####################################
# TODO recipeCategories hot encoden
####################################

print("\nEngineered DataFrames:")
print("\nDiet:")
print(diet_df.describe())
print(diet_df.columns)
print("\nRecipes:")
print(recipes_df.describe())
print(recipes_df.columns)
print("\nReviews:")
print(reviews_df.describe())
print(reviews_df.columns)
print("\nRequests:")
print(requests_df.describe())
print(requests_df.columns)
print("")

''' Splitting of the reviews.csv file into prediction, training and testing sets '''

dropColumn(reviews_df, 'Rating')
dropColumn(reviews_df, 'Like')

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

''' Regression '''

####################################
# TODO anpassen, noch kopletter murks, müssen richtige training daten auswählen
####################################

#X_train = training_df.drop('Target', axis=1)
#y_train = training_df['Target']

# Create a logistic regression model instance
#model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
#model.fit(X_train, y_train)

# Time to make predictions yikes
#X_prediction = prediction_df.drop('Target', axis=1)
#prediction_df['Predicted'] = model.predict(X_prediction)

# You can now inspect the prediction_df with the predictions
#print(prediction_df)