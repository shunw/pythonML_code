from collections import Counter

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def scatter_plot(x_ls, y_ls, classifier, resolution = .02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_ls))])

    # plot the decision surface
    x1_min, x1_max = x_ls[:, 0].min() - 1, x_ls[:, 0].max() + 1
    x2_min, x2_max = x_ls[:, 1].min() - 1, x_ls[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = .4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, c1 in enumerate(np.unique(y_ls)):
        plt.scatter(x = x_ls[y_ls == c1, 0], y = x_ls[y_ls == c1, 1], 
                    alpha = .8, c = cmap(idx), 
                    marker = markers[idx], label = c1)
    plt.show()

def multi_class_under_sampling():
    '''
    EXAMPLE: Multiclass classification with under-sampling
    '''
    RANDOM_STATE = 42

    iris = load_iris()
    X, y = make_imbalance(iris.data, iris.target, ratio = {0:25, 1:50, 2:50}, random_state = 0)
    
    # print (X[:, [1, 2]])
    # print (type(y))

    X_train, X_test, y_train, y_test = train_test_split(X[:, [1, 2]], y, random_state = RANDOM_STATE)

    # print ('Training target statistics: {}'.format(Counter(y_train)))
    # print ('Testing target statistics: {}'.format(Counter(y_test)))

    pipeline = make_pipeline(NearMiss(version = 2, random_state = RANDOM_STATE), LinearSVC(random_state = RANDOM_STATE))
    pipeline.fit(X_train, y_train)

    scatter_plot(X[:, [1, 2]], y, pipeline)
    # print (classification_report_imbalanced(y_test, pipeline.predict(X_test)))


if __name__ == '__main__':
    multi_class_under_sampling()