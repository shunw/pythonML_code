import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def data_deal():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 1)

    '''combine transformers and estimators in a pipeline'''
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components = 2)), ('clf', LogisticRegression(random_state = 1))])
    pipe_lr.fit(X_train, y_train)
    # print ('Test Accuracy: {a:.3f}'.format(a = pipe_lr.score(X_test, y_test)))


    '''using k-fold cross-validation to assess model performance'''
    # holdout cross-validation 

    # k-fold cross-validation

    # stratified k-fold cv
    kfold = StratifiedKFold(y = y_train, n_folds = 10, random_state = 1)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        # print ('Fold: {n}, Class dist.: {qty}, Acc: {score:.3f}'.format(n = k+1, qty = np.bincount(y_train[train]), score = score))
    

if __name__ == '__main__':
    data_deal()
    # stopped at page 203