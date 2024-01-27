import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

diet_df = pd.read_csv('diet.csv')
recipes_df = pd.read_csv('recipes.csv')
reviews_df = pd.read_csv('reviews.csv', low_memory=False)
requests_df = pd.read_csv('requests.csv')

print("________________________________________________________________________________")
'''##### DATA PREPARATION #####'''


def dropColumn(dataframe, column):
    if column in dataframe.columns:
        dataframe.drop(column, axis=1, inplace=True)
    else:
        print("There has been an issue with the columns in ", dataframe, ". Please check!")


'''DIET'''
# Fill NaN value in diet of the one entry that has no diet
diet_df['Diet'] = diet_df['Diet'].fillna("Vegetarian")

# Encode Diet
diet_encoded = pd.get_dummies(diet_df['Diet'], prefix='Diet', dtype=int)
diet_df = diet_df.join(diet_encoded)
dropColumn(diet_df, 'Diet')

'''RECIPES'''
# Total Time = CookTime + PrepTime
recipes_df['TotalTime'] = recipes_df['CookTime'] + recipes_df['PrepTime']
dropColumn(recipes_df, 'CookTime')
dropColumn(recipes_df, 'PrepTime')

dropColumn(recipes_df, 'Name')

# Encode RecipeCategory
recipes_encoded = pd.get_dummies(recipes_df['RecipeCategory'], prefix='RecipeCategory', dtype=int)
recipes_df = recipes_df.join(recipes_encoded)
dropColumn(recipes_df, 'RecipeCategory')

dropColumn(recipes_df, 'RecipeIngredientQuantities')  # TODO Drop for now, more intensive processing needed.
dropColumn(recipes_df, 'RecipeIngredientParts')  # TODO Drop for now, more intensive processing needed.

dropColumn(recipes_df, 'RecipeServings')  # TODO Drop for now, more intensive processing needed.
dropColumn(recipes_df, 'RecipeYield')

'''REQUESTS'''
# Replace all negative time values with absolute value
requests_df['Time'] = requests_df['Time'].apply(lambda x: abs(x) if x < 0 else x)

# Cast floats to int
requests_df['HighCalories'] = requests_df['HighCalories'].astype(int)

# Correct HighProtein column
requests_df['HighProtein'] = requests_df['HighProtein'].apply(
    lambda x: 1 if x == 'Yes' else (0 if x == 'Indifferent' else x))

dropColumn(requests_df, 'LowSugar')  # TODO Drop for now, because data seems to be nonsense
dropColumn(requests_df, 'Time')  # TODO Drop for now

'''REVIEWS'''
dropColumn(reviews_df, 'Rating')

print("\nEngineered DataFrames:")
print("\nDiet:")
print(diet_df.head())
print(diet_df.columns)
print("\nRecipes:")
print(recipes_df.head())
print(recipes_df.columns)
print("\nReviews:")
print(reviews_df.head())
print(reviews_df.columns)
print("\nRequests:")
print(requests_df.head())
print(requests_df.columns)
print("")

'''##### FEATURE ENGINEERING #####'''
recipes_df['FatCalorieProduct'] = recipes_df['FatContent'] * recipes_df['Calories']

# Standardize Content Values
columns_to_standardize = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
scaler = StandardScaler()
standardized_data = scaler.fit_transform(recipes_df[columns_to_standardize])
standardized_columns = [f'{col}_Standardized' for col in columns_to_standardize]
standardized_df = pd.DataFrame(standardized_data, columns=standardized_columns)
recipes_df = pd.concat([recipes_df, standardized_df], axis=1)
dropColumn(recipes_df, 'Calories')
dropColumn(recipes_df, 'FatContent')
dropColumn(recipes_df, 'SaturatedFatContent')
dropColumn(recipes_df, 'CholesterolContent')
dropColumn(recipes_df, 'SodiumContent')
dropColumn(recipes_df, 'CarbohydrateContent')
dropColumn(recipes_df, 'FiberContent')
dropColumn(recipes_df, 'SugarContent')
dropColumn(recipes_df, 'ProteinContent')


'''##### MERGE DATAFRAMES ######'''
merged_df = pd.merge(recipes_df, requests_df, on='RecipeId', how='inner')
merged_df = pd.merge(merged_df, diet_df, on='AuthorId', how='inner')

# reviews_df = reviews_df.dropna(subset=['Like'])

dropColumn(merged_df, 'AuthorId')
dropColumn(merged_df, 'RecipeId')

print("________________________________________________________________________________")

'''##### RUN REGRESSION ######'''

feature_columns = merged_df.columns.tolist()

print("Feature Columns: ", feature_columns)

mask = reviews_df['Like'].notna()  # Remove NA values in Like column from regression

X = merged_df[feature_columns][mask]
y = reviews_df['Like'][mask].astype(int)

print(y.value_counts()[1], "1s")
print(y.value_counts()[0], "0s")

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

print(y_resampled.value_counts()[1], "1s")
print(y_resampled.value_counts()[0], "0s")

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and fitting the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Mean CV Score:", cv_scores.mean())
