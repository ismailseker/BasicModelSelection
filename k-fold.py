from sklearn.datasets import load_iris

import pandas as pd
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

# %% normalization
x = (x - np.min(x))/(np.max(x)-np.min(x))

# %% train part
from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.3)
# %% algorithm part
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
# %% model selection cross val score
from sklearn.model_selection import cross_val_score

accurasies = cross_val_score(estimator=knn,X=xTrain,y=yTrain , cv=10)

print(np.mean(accurasies))
print(np.std(accurasies))
# %% model selection grid search cross val

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors" :np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)

print("best parameter",knn_cv.best_params_)
print("best accuracy",knn_cv.best_score_)

# %% Grid search CV with logistic regression

x = x[:100,:]
y = y[:100] 

from sklearn.linear_model import LogisticRegression

# grid = {"C":np.logspace(-3,3,7),"penalty":[["l1","l2"],'solver':['liblinear','saga']} # l1 = lasso ve l2 = ridge
grid = {"C":np.logspace(-3, 3, 7),"penalty":["l1","l2"],'solver':['liblinear','saga']}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)



