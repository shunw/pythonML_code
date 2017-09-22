from collections import Counter

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def scatter_plot(x_ls, y_ls, label_ls):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x = X[y == c1, 0], y = X[y == c1, 1], 
                    alpha = .8, c = cmap(idx), 
                    marker = markers[idx], label = c1)


def multi_class_under_sampling():
    '''
    EXAMPLE: Multiclass classification with under-sampling
    '''
    RANDOM_STATE = 42

    iris = load_iris()
    X, y = make_imbalance(iris.data, iris.target, ratio = {0:25, 1:50, 2:50}, random_state = 0)
    
    print (X[:, [0:2]])
    # print (type(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE)

    # print ('Training target statistics: {}'.format(Counter(y_train)))
    # print ('Testing target statistics: {}'.format(Counter(y_test)))

    pipeline = make_pipeline(NearMiss(version = 2, random_state = RANDOM_STATE), LinearSVC(random_state = RANDOM_STATE))
    pipeline.fit(X_train, y_train)

    # print (classification_report_imbalanced(y_test, pipeline.predict(X_test)))


if __name__ == '__main__':
    multi_class_under_sampling()