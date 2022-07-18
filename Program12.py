#Compare the models

#Random Search

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from lightgbm import LGBMClassifier

model = LGBMClassifier(random_state=0)
model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

y_predictions = model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_predictions)

#grid search

from sklearn.model_selection import GridSearchCV

parameters = {'learning_rate': [0.001, 0.01],
              'num_leaves': [2, 128],
              'min_child_samples': [1, 100],
              'subsample': [0.05, 1.0],
              'colsample_bytree': [0.1, 1.0]}

grid_search = GridSearchCV(model, parameters, n_jobs=-1)

grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print(results.sort_values(by='rank_test_score').head(3))

# picking the best model
best_model_gs = grid_search.best_estimator_

y_predictions = best_model_gs.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_predictions))




from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

parameters_distributions = {'learning_rate': loguniform(1e-3, 1e-1),
                            'num_leaves': list(range(1, 50, 5)),
                            'min_child_samples': list(range(1, 50, 5)),
                            'subsample': [0.05, 1.0],
                            'colsample_bytree': [0.1, 1.0]}

random_search = RandomizedSearchCV(model, parameters_distributions, random_state=0, n_iter=30, n_jobs=-1)

random_search.fit(X_train, y_train)

print(pd.DataFrame(random_search.cv_results_).sort_values(by='rank_test_score').head(3))

# picking the best model
best_model_rs = random_search.best_estimator_

y_predictions = best_model_rs.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_predictions)

#Compare the models
def compare_models(models, metric_function, X_test, y_test):
    scores = []
    for model in models:
        y_predictions = model.predict_proba(X_test)[:, 1]
        scores.append(metric_function(y_test, y_predictions))

    return pd.DataFrame({'models': models, 'scores': scores}).sort_values(by='scores', ascending=False)

models = [model, best_model_gs, best_model_rs]
print(compare_models(models, roc_auc_score, X_test, y_test))

