# -*- coding: utf-8 -*-
'''
To select meaningful feature: 
1. Sparse solutions with L1 regularization 
    需要进行正则化处理
2. Sequential feature selection algorithms 序列特征选择法
    通过特征选择进行降维，对未经正则化处理的模型特别有效
3. random forest tree
'''

import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def NaN_data_deal():
    '''base NaN data'''
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0'''
    csv_data = unicode(csv_data)
    df = pd.read_csv(StringIO(csv_data))
    
    '''del NaN data_ row or col'''
    a = df.dropna()
    a = df.dropna(axis = 1)
    a = df.dropna(how = 'all') # only drop rows where all col are NaN
    a = df.dropna(thresh = 4) # drop rows that have not at least 4 non-NaN values
    a = df.dropna(subset = ['C'])
    
    '''mean imputation for the NaN data'''
    imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    # print imputed_data

    '''deal the category data - ordinal feature'''
    df = pd.DataFrame([
                ['green', 'M', 10.1, 'class1'], 
                ['red', 'L', 13.5, 'class2'], 
                ['blue', 'XL', 15.3, 'class1']
    ])
    df.columns = ['color', 'size', 'price', 'classlabel']
    
    size_mapping = {
                    'XL': 3,
                    'L': 2, 
                    'M': 1
    }
    df['size'] = df['size'].map(size_mapping)
    # print df
    
    '''deal the category data - nominal feature'''
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
    
    df['classlabel'] = df['classlabel'].map(class_mapping)
    inv_map = {v: k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_map)
    
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    
    # print class_le.inverse_transform(y)
    # print y

    '''one-hot encoding in on nominal features'''
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    
    ohe = OneHotEncoder(categorical_features=[0]) # could only work on the intger
    # categorical_features = 'auto'
    # print (ohe.fit_transform(X))
    print (ohe.fit_transform(X).toarray())
    
    print pd.get_dummies(df[['price', 'color', 'size']]) # could only work on the string

def L1_regularization():
    df_wine = pd.read_csv('wine.data', header = None)    
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    # print('Class labels', np.unique(df_wine['Class label']))
    # print df_wine.head()

    '''partition dataset with train and test'''
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

    '''bring feature on to the same scale'''
    # use normalization to scale
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    # use the standardization to scale
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    # print (y_train)
    X_test_std = stdsc.transform(X_test)

    ''' to support L1 regularization, penalty parameter set to '11' to yield sparse solution'''
    LogisticRegression(penalty = 'l1')
    lr = LogisticRegression(penalty = 'l1', C = .1)
    lr.fit(X_train_std, y_train)
    print ('Training accuracy: ', lr.score(X_train_std, y_train))
    print ('Test accuracy: ', lr.score(X_test_std, y_test))
    print lr.intercept_
    print lr.coef_

    '''plot regularization path'''
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
    weights, params = [], []
    for c in np.arange(-4, 6): 
        lr = LogisticRegression(penalty = 'l1', C = 10 ** c, random_state = 0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10 ** c)
    # print weights
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        # weights.shape[1] is the 13 features of a wine
        plt.plot(params, weights[:, column], 
                label = df_wine.columns[column + 1], 
                color = color)

    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
    plt.xlim([10 ** (-5), 10 ** 5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc = 'upper left')
    ax.legend(loc = 'upper center', bbox_to_anchor = (1.38, 1.03), ncol = 1, fancybox = True)
    plt.show()

class SBS():
    # sbs is sequential backward selection
    def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = .25, random_state = 1):
        self.scoring = scoring 
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
        
        # this part get the initiate one, and get the score on all the features
        dim = X_train.shape[1] # get the total feature number
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r = dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def seq_feature_select():
    # df_wine = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wine/wine.data', header=None)  
    df_wine = pd.read_csv('wine.data', header=None)  
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    '''sbs'''
    knn = KNeighborsClassifier(n_neighbors = 2)
    sbs = SBS(knn, k_features = 1)
    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker = 'o')
    plt.ylim([0.5, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    # plt.show()

    # get which 5 features are the key features
    k5 = list(sbs.subsets_[8])
    print (df_wine.columns[1:][k5])

    # validate knn in the original test data
    knn.fit(X_train_std, y_train)
    print ('Training accuracy: ', knn.score(X_train_std, y_train))
    print ('Test accuracy: ', knn.score(X_test_std, y_test))

    knn.fit(X_train_std[:, k5], y_train)
    print ('Training accuracy: ', knn.score(X_train_std[:, k5], y_train))
    print ('Test accuracy: ', knn.score(X_test_std[:, k5], y_test))

def random_forest(): 
    df_wine = pd.read_csv('wine.data', header=None)  
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)
    
    # no need to do the normalization/ standarization
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    # print importances
    indices = np.argsort(importances)[::-1]
    # print np.argsort(importances)
    # print indices
    # for f in range(X_train.shape[1]):
    #     print ("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    '''bar plot for the importance'''
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], color = 'lightblue', align = 'center')
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    # plt.show()

    '''set threshold'''
    X_selected = forest.transform(X_train, threshold = .15)
    print X_selected.shape
    # print X_selected


if __name__ == '__main__':
    # L1_regularization()

    # seq_feature_select()

    random_forest()
    
