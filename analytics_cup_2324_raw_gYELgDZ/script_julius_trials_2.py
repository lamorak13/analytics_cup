import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, \
    precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

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
recipes_df['ProteinFiberProduct'] = recipes_df['ProteinContent'] * recipes_df['FiberContent']
recipes_df['TotalTime^2'] = recipes_df['TotalTime'] * recipes_df['TotalTime']

# Standardize Content Values
columns_to_standardize = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
                          'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
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

# Categorize RecipeIngredientParts
omnivore_keywords = [
    "Beef", "Chicken", "Pork", "Lamb", "Turkey", "Veal", "Bacon", "Sausage", "Steak", "Ground beef",
    "Ham", "Ribs", "Venison", "Duck", "Quail", "Bison", "Meatballs", "Goat", "Rabbit", "Goose",
    "Seafood", "Shrimp", "Salmon", "Tuna", "Cod", "Lobster", "Crab", "Clams", "Oysters", "Mussels",
    "Scallops", "Tilapia", "Catfish", "Swordfish", "Haddock", "Halibut", "Mahi-mahi", "Octopus", "Squid", "Eel",
    "Anchovies", "Mackerel", "Herring", "Trout", "Snapper", "Grouper", "Flounder", "Perch", "Sole", "Caviar",
    "Prosciutto", "Parma ham", "Chorizo", "Bratwurst", "Corned beef", "Pastrami", "Pate", "Salami", "Pepperoni",
    "Kielbasa", "Liver", "Kidneys", "Heart", "Tongue", "Tripe", "Haggis", "Blood sausage", "Chitterlings",
    "Crayfish", "Prawn", "Monkfish", "Sea bass", "Sardines", "Pike", "Carp", "Arctic char", "King crab", "Snow crab",
    "Conch", "Sea urchin", "Abalone", "Lumpfish", "Pufferfish", "Swordfish", "Bluefish", "Pompano", "Redfish",
    "Scrod", "Shark", "Sturgeon", "Tilefish", "Wahoo", "Yellowtail", "Frog legs", "Snails", "Alligator", "Kangaroo",
    "Wild boar", "Ostrich", "Guinea fowl", "Pheasant", "Partridge", "Emu", "Camel", "Elk", "Moose", "Reindeer",
    "Bear", "Buffalo", "Crocodile", "Rattlesnake", "Turtle", "Pigeon", "Dove", "Squirrel", "Porcupine", "Raccoon"
]

vegetarian_keywords = [
    "Cheese - Cheddar", "Cheese - Mozzarella", "Cheese - Parmesan",
    "Eggs - Chicken", "Eggs - Duck", "Eggs - Quail",
    "Butter",
    "Milk - Cow", "Milk - Goat", "Milk - Buffalo",
    "Yogurt", "Greek Yogurt",
    "Cream - Heavy", "Cream - Light", "Sour Cream",
    "Paneer",
    "Ghee",
    "Honey",
    "Gelatin (non-vegetarian but used in some vegetarian cuisines)",
    "Casein",
    "Whey Protein",
    "Custard (made with eggs)",
    "Mayonnaise (traditional recipes contain egg)",
    "Quiche",
    "Frittata",
    "Ricotta Cheese",
    "Brie Cheese",
    "Camembert Cheese",
    "Feta Cheese",
    "Halloumi Cheese",
    "Mascarpone",
    "Tzatziki (made with yogurt)",
    "Lassi (yogurt-based drink)",
    "Ice Cream",
    "Cottage Cheese",
    "Cream Cheese",
    "Sour Cream",
    "Egg Noodles",
    "Meringue (made with egg whites)",
    "Creme Brulee (contains cream and eggs)",
    "Pavlova (meringue-based dessert)",
    "Eggnog (made with milk and eggs)",
    "Caramel (dairy-based)",
    "Panna Cotta (contains gelatin and cream)",
    "Flan (contains eggs and milk)",
    "Tiramisu (contains mascarpone cheese, cream, and sometimes egg)",
    "Chocolate (milk chocolate)",
    "Buttermilk",
    "Condensed Milk",
    "Evaporated Milk",
    "Alfredo Sauce (contains cream and cheese)",
    "Cheese - Gouda", "Cheese - Swiss", "Cheese - Blue Cheese",
    "Cheese - Roquefort", "Cheese - Colby", "Cheese - Monterey Jack",
    "Cheese - Havarti", "Cheese - Provolone", "Cheese - Edam",
    "Cheese - Emmental", "Cheese - Gorgonzola", "Cheese - Stilton",
    "Cheese Spread", "Cheese Fondue",
    "Egg Salad", "Eggplant Parmesan",
    "Omelette",
    "Dairy-Based Salad Dressing",
    "Milkshake",
    "Vegetarian Pudding",
    "Cheese Pizza", "Vegetarian Lasagna",
    "Vegetarian Sausage (contains eggs and/or dairy)",
    "Egg Custard",
    "French Toast (made with eggs and milk)",
    "Grilled Cheese Sandwich",
    "Macaroni and Cheese",
    "Vegetarian Quorn products (some contain egg and/or dairy)",
    "Scrambled Eggs",
    "Souffle (contains eggs and/or cheese)",
    "Veggie Omelet",
    "Yogurt Parfait",
    "Custard Tart (contains eggs and milk)",
    "Dulce de Leche (milk-based)",
    "Vegetarian Gelato",
    "Milk-based Indian Sweets (like Rasgulla, Rasmalai)",
    "Protein Bars (some contain dairy or eggs)",
    "Whey-based Protein Shakes",
    "Egg-based Pasta",
    "Cheese Croissant",
    "Cheese Strudel",
    "Egg-based Breakfast Burrito",
    "Milk-based Soups and Sauces",
    "Cheese Blintz",
    "Yogurt Dressing",
    "Butter Toffee",
    "Cheese Souffle",
    "Egg Drop Soup",
    "Egg Fried Rice",
    "Greek Salad with Feta",
    "Mozzarella Sticks",
    "Paneer Tikka",
    "Ricotta Pancakes",
    "Spinach and Cheese Ravioli",
    "Vegetarian Caesar Salad (traditional recipe contains anchovies)",
    "Cheese and Spinach Pie",
    "Egg Bhurji (Indian scrambled eggs)",
    "Yogurt Smoothie",
    "Vegetarian Cheeseburger (with cheese and vegetarian patty)",
    "Cream-based Pastas",
    "Quark (fresh dairy product)",
    "Labneh (strained yogurt)",
    "Kefir (fermented milk drink)",
    "Vegetarian Pate (made with cheese or eggs)",
    "Deviled Eggs",
    "Egg Salad Sandwich",
    "Egg-based Breads and Pastries",
    "Milk Tea",
    "Vegetarian Egg Rolls (with dairy or egg-based fillings)",
    "Cheese Omelet",
    "Cottage Cheese and Fruit",
    "Dairy-Based Smoothies",
    "Milk-Based Protein Powder",
    "Vegetarian Shepherd's Pie (with dairy and/or egg ingredients)"
]


def classify_ingredient(ingredient):
    ingredient_lower = ingredient.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    if any(keyword.lower() in ingredient_lower for keyword in omnivore_keywords):
        return 'omnivore'
    elif any(keyword.lower() in ingredient_lower for keyword in vegetarian_keywords):
        return 'vegetarian'
    else:
        return 'vegan'


# Apply classification function to the 'RecipeIngredientParts' column
recipes_df['RecipeIngredientPartsClassification'] = recipes_df['RecipeIngredientParts'].apply(classify_ingredient)
dropColumn(recipes_df, 'RecipeIngredientParts')

recipes_encoded = pd.get_dummies(recipes_df['RecipeIngredientPartsClassification'], prefix='RecipeIngredientPartsClassification', dtype=int)
recipes_df = recipes_df.join(recipes_encoded)
dropColumn(recipes_df, 'RecipeIngredientPartsClassification')


'''##### MERGE DATAFRAMES ######'''
merged_df = pd.merge(requests_df, diet_df, on='AuthorId', how='left')
merged_df = pd.merge(merged_df, recipes_df, on='RecipeId', how='left')
print(len(merged_df))

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

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of the SMOTE class
smote = SMOTE(sampling_strategy='auto', random_state=2024)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
X_train_scaled = X_train_resampled
y_train = y_train_resampled

# Creating and fitting the model

# XGBoost
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#model.fit(X_train_scaled, y_train)
#y_probs = model.predict_proba(X_test_scaled)[:, 1]
#threshold = 0.18
#y_pred = (y_probs >= threshold).astype(int)

# RandomForest
model_rf = RandomForestClassifier(n_estimators=100, random_state=2024)
#model.fit(X_train_scaled, y_train)
#y_probs = model.predict_proba(X_test_scaled)[:, 1]
#threshold = 0.3
#y_pred = (y_probs >= threshold).astype(int)

# Logistic
model_logistic = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced')
#model.fit(X_train_scaled, y_train)
#y_pred = model.predict(X_test_scaled)

# LightGBM
model_lgb = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
#model.fit(X_train_scaled, y_train)
#y_probs = model.predict_proba(X_test_scaled)[:, 1]
#threshold = 0.18
#y_pred = (y_probs >= threshold).astype(int)

# Create a Voting Classifier
model = VotingClassifier(estimators=[
    ('xgb', model_xgb),
    ('rf', model_rf),
    ('logistic', model_logistic),
    ('lgb', model_lgb)
], voting='soft')

# Fit the ensemble model
model.fit(X_train_scaled, y_train)

# Make predictions using the ensemble model
y_probs = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.29
y_pred = (y_probs >= threshold).astype(int)


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

# Extracting True Negatives, False Positives, False Negatives, and True Positives
tn, fp, fn, tp = confusion_matrix.ravel()

# Calculating Sensitivity and Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
balanced_accuracy = (sensitivity + specificity) / 2

print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Balanced Accuracy: {balanced_accuracy}")

# Perform cross-validation
#cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
#print("Mean CV Score:", cv_scores.mean())


# Filter the rows where 'Like' is NA
predict_df = reviews_df[reviews_df['Like'].isna()]

# Select the same feature columns used in the model
X_predict = merged_df.loc[predict_df.TestSetId, feature_columns]

# Predict Using the Model
X_predict_scaled = scaler.transform(X_predict)

# Make predictions
y_probs_2 = model.predict_proba(X_predict_scaled)[:, 1]
y_pred_2 = (y_probs_2 >= threshold).astype(int)
predictions = y_pred_2

# Create a New DataFrame for Predictions
predictions_df = pd.DataFrame({
    'id': predict_df['TestSetId'],
    'prediction': predictions
})

if predictions_df['id'].dtype == 'float':
    predictions_df['id'] = predictions_df['id'].astype(int)

# Write to CSV
predictions_df.to_csv('predictions_die_bummler_1.csv', index=False)
print("Written to CSV.")