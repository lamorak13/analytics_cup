import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


def print_outliers(df, attribute, threshold):
    z = np.abs(stats.zscore(df[attribute]))
    outliers = df[(z > threshold)]
    print(outliers)


def remove_outliers(df, attribute, threshold) -> pd.DataFrame:
    z = np.abs(stats.zscore(df[attribute]))
    return df[(z < threshold)]


def replace_outliers(df: pd.DataFrame, attribute, threshold, value) -> pd.DataFrame:
    z = np.abs(stats.zscore(df[attribute]))
    df.loc[z > threshold, attribute] = value
    return df


def replace_negative_values(df: pd.DataFrame, attribute, value) -> pd.DataFrame:
    df.loc[df[attribute] < 0, attribute] = value
    return df


# Load data
diet_df = pd.read_csv("diet.csv")
recipes_df = pd.read_csv("recipes.csv")
reviews_df = pd.read_csv("reviews.csv", low_memory=False)
requests_df = pd.read_csv("requests.csv")


# ---------------------------- Data Preparation -------------------------
# --- Diet
diet_df["Diet"].fillna("Vegetarian", inplace=True)

# --- Recipes
""" recipes_df = recipes_df.drop("RecipeYield", inplace=True, axis=1)
recipes_df["TotalTime"] = recipes_df["CookTime"] + recipes_df["PrepTime"]
reciped_df = remove_outliers(recipes_df, "TotalTime", 3.5)
reciped_df = pd.get_dummies(data=reciped_df, columns=["RecipeCategory"])

recipes_df["RecipeServings"].fillna(recipes_df["RecipeServings"].median(), inplace=True)
reciped_df = remove_outliers(recipes_df, "RecipeServings", 3.5)

# --- Reviews
reviews_df.drop("Rating", inplace=True, axis=1)
unlabeled_dataset = reviews_df[reviews_df["TestSetId"] != "Nan"]
labeled_dataset = reviews_df[reviews_df["TestSetId"] == "Nan"] """

# --- Requests
""" requests_df = replace_outliers(
    requests_df, "Time", 3.5, requests_df["Time"].quantile(0.5)
)
requests_df = replace_negative_values(
    requests_df, "Time", requests_df["Time"].quantile(0.25)
) """

ingredients_set = set()

for x in recipes_df["RecipeIngredientParts"]:
    ingredients = x[2:-1].replace('"', "").replace("\\", "").split(",")
    for i in ingredients:
        ingredients_set.add(i.strip())

print(ingredients_set)
