import pandas as pd

#Grid Search
#hyperparameter values when set right can build highly accurate
#models
#We have three methods of hyperparameter tuning in python are Grid search, Random search, and Informed search.

#Grid Search

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset)

#Create matrix of independent variables
X = dataset.iloc[:,[2,3]].values #[:,1] would return an array but we want X to be a
#matrix

#Create array of dependent variable
y = dataset.iloc[:,4].values

#Splitting into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

#Predicting the test set value
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies)

print(accuracies.mean())

print(accuracies.std())

