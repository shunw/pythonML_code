import matplotlib.pyplot as plt
import math
import numpy as np
import operator
from scipy.misc import comb
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone  
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import six
from sklearn.linear_model import LogisticRegression
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
        print ('Accuracy: {score_mean:.2f} (+/- {score_std:.2f}) [{label}]'.format(score_mean = scores.mean(), score_std = scores.std(), label = label))

        
if __name__ == '__main__':
    # print ensemble_error(n_classifier = 11, error = .25)
    # ensembles_errors()
    simple_majority_vote()