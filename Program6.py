#hyperparameter tuning in python are Random search,

#https://www.kaggle.com/code/pavansanagapati/automated-hyperparameter-tuning/data?select=creditcard.csv
import pandas as pd
wine_df = pd.read_csv("winequality-red.csv")
print(wine_df.head())
#Any Missing rows
print(wine_df.isna().sum() / len(wine_df))

