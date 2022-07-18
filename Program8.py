#hyperparameter tuning in python are Random search,

#https://www.kaggle.com/code/pavansanagapati/automated-hyperparameter-tuning/data?select=creditcard.csv
import pandas as pd
import numpy as np
wine_df = pd.read_csv("winequality-red.csv")
print(wine_df.head())
#Any Missing rows
print(wine_df.isna().sum() / len(wine_df))


wine_df = wine_df.assign(good_wine=lambda df: np.where(df.quality < 7, 0, 1))
print(wine_df)

#Change to one value. yes/No


print((
    wine_df
    .groupby('good_wine')
    .agg(n=('good_wine', 'size'))
    .reset_index()
    .assign(n_prop=lambda df: 100 * (df.n / df.n.sum()))
))