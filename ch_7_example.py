from itertools import product
import matplotlib.pyplot as plt
import math
import numpy as np
import operator
import pandas as pd
from scipy.misc import comb
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone  
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import six
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import _name_estimators
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error ** k * (1 - error) ** (n_classifier - k) for k in range(int(k_start), n_classifier + 1)]
    return sum(probs)

def ensembles_errors():
    '''learning with ensembles'''
    
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier = 11, error = error) for error in error_range]
    plt.plot(error_range, ens_errors, label = 'Ensemble error', linewidth = 2)
    plt.plot(error_range, error_range, linestyle = '--', label = 'Base error', linewidth = 2)
    plt.xlabel('Base error')
    plt.ylabel('Base/Ensemble error')
    plt.legend(loc = 'upper left')
    plt.grid()
    plt.show()

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin): 
    ''' A majority vote ensemble classifier
    Parameters 
    ---------------------
    classifiers: array-like, shape = [n_classifiers] 
        Different classifiers for the ensemble

    vote: str, {'classlabel', 'probability'}
        Default: 'classlabel'
        If 'classlabel' the prediction is based on the argmax of class labels. Else if 'probability', the argmax of the sum of probabilities is used to predict the class label (recommended for calibrated classifiers). 

    weights: array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of 'int' or 'float' values are provided, the classifiers are weighted by importance; Uses uniform weights if 'weights = None'.
    '''
    def __init__(self, classifiers, vote = 'classlabel', weights = None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self, X, y):
        '''Fit classifiers. 
        Parameters
        --------------
        X: {array-like, sparse matrix}, 
            shape = [n_samples, n_features]
            Matrix of training samples

        y: array-like, shape = [n_samples]
            Vector of target class labels. 
        
        Returns
        --------------
        self: object
        '''
        # Use LabelEncoder to ensure class labels start with 0, which is important for np.argmax call in self.predict

        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers: 
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        '''Predict class labels for X

        Parameters
        ----------
        X: {array-like, sparse matrix},
            Shape = [n_samples, n_features]
            Matrix of training samples
        
        Returns
        ----------
        maj_vote: array-like, shape = [n_samples]
            Predictd class labels

        '''
        if self.vote == 'probability':
            maj_vote == np.argmax(self.predict_proba(X), axis = 1)
        else: #'classlabel' vote
            #collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)), axis = 1, arr = predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
            return maj_vote
        
    def predict_proba(self, X):
        '''Predict class probabilities for X
        
        Paramters
        -----------
        X: {array-like, sparse matrix}, 
            shape = [n_sampels, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features. 

        Returns
        -----------
        avg_proba: array-like
            shape = [n_samples, n_classes]
            Weigthed average probability for each class per sample.

        '''
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)
        return avg_proba

    def get_params(self, deep = True):
        '''Get classifier paramter names for GridSearch'''
        if not deep: 
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep = True)):
                    out['{name}__{key}'.format(name = name, key = key)] = value
            return out

def iris_data():
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = .5, random_state = 1)
    return X_train, X_test, y_train, y_test

def simple_majority_vote():
    # weighted multi vote 
    # print np.argmax(np.bincount([0, 0, 1], weights = [.2, .2, .6]))

    ex = np.array([[.9, .1], [.8, .2], [.4, .6]])
    p = np.average(ex, axis = 0, weights = [.2, .2, .6])
    # print p
    # print np.argmax(p)

    X_train, X_test, y_train, y_test = iris_data()

    clf1 = LogisticRegression(penalty = 'l2', C = .001, random_state = 0)
    clf2 = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy', random_state = 0)
    clf3 = KNeighborsClassifier(n_neighbors = 1, p = 2, metric = 'minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    # print ('10-fold cross validation: \n')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, scoring = 'roc_auc')
        # print ('ROC AUC: {score_mean:.2f} (+/- {score_std:.2f}) [{label}]'.format(score_mean = scores.mean(), score_std = scores.std(), label = label))

    mv_clf = MajorityVoteClassifier(classifiers = [pipe1, clf2, pipe3])
    clf_labels += ['Majority Voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    for clf, label in zip(all_clf, clf_labels): 
        scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, scoring = 'roc_auc')
        # print ('Accuracy: {score_mean:.2f} (+/- {score_std:.2f}) [{label}]'.format(score_mean = scores.mean(), score_std = scores.std(), label = label))

    # # plot the ROC comparision plot
    # colors = ['black', 'orange', 'blue', 'green']
    # linestyles = [':', '--', '-.', '-']
    # for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    #     # assume the label of the positive class is 1
    #     y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    #     fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_pred)
    #     roc_auc = auc(x = fpr, y = tpr)
    #     plt.plot(fpr, tpr, color = clr, linestyle = ls, label = '{label} (auc = {roc_auc:.2f})'.format(label = label, roc_auc = roc_auc))
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', linewidth = 2)
    # plt.xlim([-.1, 1.1])
    # plt.ylim([-.1, 1.1])
    # plt.grid()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    # # plot the decision region
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train)
    # x_min = X_train_std[:, 0].min() - 1
    # x_max = X_train_std[:, 0].max() + 1
    # y_min = X_train_std[: ,1].min() - 1
    # y_max = X_train_std[: ,1].max() + 1

    # xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    # f, axarr = plt.subplots(nrows = 2, ncols = 2, sharex = 'col', sharey = 'row', figsize = (7, 5))

    # for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    #     clf.fit(X_train_std, y_train)
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #     axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha = .3)
    #     axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0], X_train_std[y_train == 0, 1], c = 'blue', marker = '^', s = 50)
    #     axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], c = 'red', marker = 'o', s = 50)
    #     axarr[idx[0], idx[1]].set_title(tt)
    # plt.text(-3.5, -4.5, s = 'Sepal width [standardized]', ha = 'center', va = 'center', fontsize = 12)
    # plt.text(-10.5, 4.5, s = 'Petal length [standardized]', ha = 'center', va = 'center', fontsize = 12, rotation = 90)
    # plt.show()

    # page 216/241
    # print mv_clf.get_params()
    params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [.001, .1, 100.0]}
    grid = GridSearchCV(estimator = mv_clf, param_grid = params, cv = 10, scoring = 'roc_auc')
    grid.fit(X_train, y_train)

    for params, mean_score, scores in grid.grid_scores_:
        print ('{mean_score:3f} +/- {scores_std:2f} {params}'.format(mean_score = mean_score, scores_std = scores.std()/2, params = params))

def bagging_sample():
    df_wine = pd.read_csv('wine.data', header = None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD289/OD315 of diluted wines', 'Proline']
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'Hue']].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 1)

    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = 1)
    bag = BaggingClassifier(base_estimator = tree, n_estimators = 500, max_samples = 1.0, max_features = 1.0, bootstrap = True, bootstrap_features = False, n_jobs = 1, random_state = 1)

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print ('Decision tree train/ test accuracies {tree_train:.3f}/{tree_test:.3f}'.format(tree_train = tree_train, tree_test = tree_test))

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)
    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print ('Bagging train/ test accuracies {bag_train:.3f}/{bag_test:.3f}'.format(bag_train = bag_train, bag_test = bag_test))

    # x_min = X_train[:, 0].min() - 1
    # x_max = X_train[:, 0].max() + 1
    # y_min = X_train[:, 1].min() - 1
    # y_max = X_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    # f, axarr = plt.subplots(nrows = 1, ncols = 2, sharex = 'col', sharey = 'row', figsize = (8, 3))
    # # print ('X_train is {x_train}'.format(x_train = X_train[y_train == 0]))
    # # print ('y_train is {y_train}'.format(y_train = y_train))
    # for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    #     clf.fit(X_train, y_train)

    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #     # print ('idx: {idx}: X.shape is {x_shape}'.format(idx = idx, x_shape = X_train[y_train == 1].shape))
    #     # print (X_train[y_train == 0])
    #     axarr[idx].contourf(xx, yy, Z, alpha = .3)
    #     axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c = 'blue', marker = '^', alpha = .9)
    #     axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c = 'red', marker = 'o', alpha = .9)
    #     axarr[idx].set_title(tt)

    # axarr[0].set_ylabel('Alcohol', fontsize = 12)
    # plt.text(10.2, -0.75, s = 'Hue', ha = 'center', va = 'center', fontsize = 12)
    # plt.show()

def ada_boost():
    df_wine = pd.read_csv('wine.data', header = None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD289/OD315 of diluted wines', 'Proline']
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'Hue']].values
    # X = df_wine.iloc[:, 1:].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 1)

    tree = DecisionTreeClassifier(criterion = "entropy", max_depth = None, random_state = 0)
    ada = AdaBoostClassifier(base_estimator = tree, n_estimators = 500, learning_rate = .1, random_state = 0)
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print ('Decision tree train/test accuracies {tree_train:.3f} / {tree_test:.3f}'.format(tree_train = tree_train, tree_test = tree_test))

    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print ('AdaBoost train/test accuracies {ada_train:.3f} / {ada_test:.3f}'.format(ada_train = ada_train, ada_test = ada_test))

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    f, axarr = plt.subplots(1, 2, sharex = 'col', sharey = 'row', figsize = (8, 3))
    for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha = .3)
        axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c = 'red', marker = 'o')
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize = 12)
    plt.text(10.2, -.75, s = 'Hue', ha = 'center', va = 'center', fontsize = 12)
    plt.show()


if __name__ == '__main__':
    # print ensemble_error(n_classifier = 11, error = .25)
    # ensembles_errors()

    # simple_majority_vote()
    # bagging_sample()
    ada_boost()
    # page 238/ 213