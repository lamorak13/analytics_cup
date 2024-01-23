import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


# Load data
# diet_df = pd.read_csv("diet.csv")
# recipes_df = pd.read_csv("recipes.csv")
reviews_df = pd.read_csv("reviews.csv", low_memory=False)
# requests_df = pd.read_csv("requests.csv")


""" column = "TestSetId"
maske = reviews_df[column].isnull()
print("---------------- head -------------------")
print(reviews_df[column].head(20))
print("---------------- describe -------------------")
print(reviews_df[column].describe())
print("---------------- value types -------------------")
print(reviews_df[column].dtypes)
print("---------------- null values -------------------")
print(reviews_df[maske])
print(reviews_df[maske].shape)
print(reviews_df[column].unique())
print(reviews_df[column].value_counts()) """


unlabeled_dataset = reviews_df[reviews_df["TestSetId"] != "Nan"]
labeled_dataset = reviews_df[reviews_df["TestSetId"] == "Nan"]
