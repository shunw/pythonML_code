# -*- coding: utf-8 -*-
'''
perceptron algorithm 
    to minimize misclassification error
SVM (support vector machine)
    maxmize the margin (margin is defined as the distance between the separating hyperplane, and the training samples that are closest to this hyperplane)
'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import numpy as np


def scikit_basic():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(n_iter = 40, eta0 = .1, random_state = 0)
    ppn.fit(X_train_std, y_train)
    
    y_pred = ppn.predict(X_test_std)
    # print ('Misclassified samples: %d' % (y_test != y_pred).sum())
    # print ('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X = X_combined_std, y = y_combined, classifier = ppn, test_idx = range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = .02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = .4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], 
                    alpha = .8, c = cmap(idx), 
                    marker = markers[idx], label = cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c = '',
                     alpha = 1.0, linewidth = 1, marker = 'o', 
                     s = 55, label = 'test set')
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

class iris_data:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data[:, [2, 3]]
        self.y = iris.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = .3, random_state = 0)
    def norm_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def std_data(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        return self.X_train_std, self.y_train, self.X_test_std, self.y_test
    
    def combined_data(self):
        self.std_data()
        X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        y_combined = np.hstack((self.y_train, self.y_test))
        return X_combined_std, y_combined

def logistic_base():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))


    ''' logistic plot '''
    # z = np.arange( -7, 7, .1)
    # phi_z = sigmoid(z)
    # plt.plot(z, phi_z)
    # plt.axvline(0.0, color = 'k')
    # plt.axhspan(0.0, 1.0, facecolor = '1.0', alpha = 1.0, ls = 'dotted')
    # plt.axhline(y = .5, ls = 'dotted', color = 'k')
    # plt.yticks([0.0, .5, 1.0])
    # plt.ylim(-.1, 1.1)
    # plt.xlabel('z')
    # plt.ylabel('$\phi (z)$')
    # plt.show()

    ''' fit iris with logistic'''
    # lr = LogisticRegression(C = 1000.0, random_state = 0)
    # lr.fit(X_train_std, y_train)
    # plot_decision_regions(X_combined_std, y_combined, classifier = lr, test_idx = range(105, 150))
    # plt.xlabel('petal length [standardized]')
    # plt.ylabel('petal width [standardized]')
    # plt.legend(loc = 'upper left')
    # plt.show()

    # print (lr.predict_proba(X_test_std[0, :]))

    '''add L2 regularization'''
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C = 10 ** c, random_state = 0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        print (lr.coef_[1])
        params.append(10 ** c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label = 'petal length')
    plt.plot(params, weights[:, 1], linestyle = '--', label = 'petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc = 'upper left')
    plt.xscale('log')
    plt.show()


def SVM_related():
    '''solve linear problem'''
    svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
    ir = iris_data()
    X_train_std, y_train, X_test_std, y_test = ir.std_data()
    X_combined_std, y_combined = ir.combined_data()
    svm.fit(X_train_std, y_train)

    # plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105, 150))
    # plt.xlabel('petal length [standardized]')
    # plt.ylabel('petal width [standardized]')
    # plt.legend(loc = 'upper left')
    # plt.show()

    '''solve non-linear problem'''
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    # plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c = 'b', marker = 'x', label = '1')
    # plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c = 'r', marker = 's', label = '-1')
    # plt.ylim(-3.0)
    # plt.legend()
    # plt.show()

    # svm = SVC(kernel = 'rbf', random_state = 0, gamma = .10, C = 10.0)
    # svm.fit(X_xor, y_xor)
    # plot_decision_regions(X_xor, y_xor, classifier = svm)
    # plt.legend(loc = 'upper left')
    # plt.show()

    # for better understanding on the gamma parameter
    # svm = SVC(kernel = 'rbf', random_state = 0, gamma = .2, C = 1.0)
    # svm.fit(X_train_std, y_train)
    # plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105, 150))
    # plt.xlabel('petal length [standarized]')
    # plt.ylabel('petal width [standarized]')
    # plt.legend(loc = 'upper left')
    # plt.show()


def decision_tree_related():
    ir = iris_data()
    X_train, y_train, X_test, y_test = ir.norm_data()
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
    tree.fit(X_train, y_train)

    plot_decision_regions(X_combined, y_combined, classifier = tree, test_idx = range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.show()

def random_forest_related():
    ir = iris_data()
    X_train, y_train, X_test, y_test = ir.norm_data()
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    forest = RandomForestClassifier(criterion = "entropy", n_estimators = 10, random_state = 1, n_jobs = 2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier = forest, test_idx = range(105, 150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc = 'upper left')
    plt.show()
    
def k_nearest():
    ir = iris_data()
    X_train, y_train, X_test, y_test = ir.norm_data()
    X_train_std, y_train, X_test_std, y_test = ir.std_data()
    X_combined_std, y_combined = ir.combined_data()
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier = knn, test_idx = range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.show()


if __name__ == '__main__':
    # scikit_basic()

    k_nearest()

    