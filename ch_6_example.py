import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def data_deal():
    df = pd.read_csv('wdbc.data', header=None)

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
    
    scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)
    # print ('CV accruracy scores: {scores}'.format(scores = scores))
    # print ('CV accuracy: {score_mean:.3f} +/- {score_std:.3f}'.format(score_mean = np.mean(scores), score_std = np.std(scores)))

    '''diagnosing bias and variace with learning curves'''
    pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty = 'l2', random_state = 0))])
    train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X = X_train, y = y_train, train_sizes = np.linspace(.1, 1.0, 10), cv = 10, n_jobs = 1)
    
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = .15, color = 'blue')
    plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = .15, color = 'green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right')
    plt.ylim([.8, 1.0])
    plt.show()

if __name__ == '__main__':
    data_deal()
    # stopped at page 183/ 208