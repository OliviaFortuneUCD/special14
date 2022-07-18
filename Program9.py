#hyperparameter tuning in python are Random search,

#https://www.kaggle.com/code/pavansanagapati/automated-hyperparameter-tuning/data?select=creditcard.csv
import pandas as pd
import numpy as np
wine_df = pd.read_csv("winequality-red.csv")
print(wine_df.head())
#Any Missing rows
print(wine_df.isna().sum() / len(wine_df))


wine_df = wine_df.assign(good_wine=lambda df: np.where(df.quality < 7, 0, 1))


from sklearn.model_selection import train_test_split
X = wine_df.drop(['quality', 'good_wine'], axis='columns')
y = wine_df['good_wine']
print(X.head())
print(y.head())

