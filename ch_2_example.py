import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed

class perceptron(object):
    ''' perceptron classifier.
    Parameters
    -----------------------
    eta: float
        learning rate (between 0.0 and 1.0)
    n_iter: int
        passes over the training dataset. 

    Attributes
    -----------------------
    w_: ld-array
        weights after fitting.
    errors_: list
        number of misclassifications in every epoch. 
    '''
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''Fit training data.

        Parameters
        -----------------------
        X: {array-like}, shape = [n_samples, n_features]
            training vectors, where 
            n_samples is the number of samples and 
            n_features is the number of features. 
        y: array-like, shape = [n_samples]
            target values. 
        
        Returns
        -----------------------
        self: object
        '''

        self.w_ = np.zeros(1 + X.shape[1])  
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''return class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class adalineGD(object):
    '''adaptive linear neuron classifier. 

    Parameters 
    -----------------------
    eta: float
        learning rate (between 0.0 and 1.0)
    n_iter: int
        passes over the training dataset. 

    Attributes
    -----------------------
    w_: 1d-array
        weights after fitting.
    errors_: list
        number of misclassifications in every epoch. 

    '''
    def __init__(self, eta = .01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        ''' fit training data.

        Parameters
        -----------------------
        X: {array - like}, shape = [n_samples, n_features]
            training vectors, 
            where n_samples is the number of samples and 
            n_features is the number of features. 
        y: array-like, shape = [n_samples]
            target values.

        Returns 
        -----------------------
        self: object
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        ''' calculate net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        ''' compute linear activation '''
        return self.net_input(X)
    
    def predict(self, X):
        ''' return class label after unit step'''
        return np.where(self.activation(X) >= 0.0, 1, -1)

class adalineSGD(object):
    '''adaptive linear neuron classifier.

    Parameters
    -----------------------
    eta: float
        learning rate (between 0.0 and 1.0)
    n_iter: int
        passes over the training dataset.

    Attributes
    -----------------------
    w_: 1d-array
        weights after fitting. 
    errors_: list
        number of misclassifications in every epoch. 
    shuffle: bool (default: True)
        shuffles training data every epoch
        if True to prevent cycles.
    random_state: int (default: None)
        set random state for shuffling
        and initializing the weights. 
    '''
    def __init__(self, eta = .01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state: seed(random_state)

    def fit(self, X, y):
        '''fit training data. 

        Parameters
        -----------------------
        X: {array-like}, shape = [n_samples, n_features]
            training vectors, where n_samples is the number of samples and n_features is the number of features. 

        y: array-like, shape = [n_samples]
            target values. 
        
        Returns 
        -----------------------
        self: object
        '''
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle: 
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        ''' fit training data without reinitializing the weights'''
        if not self.w_initialized: 
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else: 
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        ''' shuffle training data'''
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        ''' initialize weights to zeors'''
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        ''' apply adaline learning rule to update the weights'''
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = .5 * error ** 2
        return cost 

    def net_input(self, X):
        '''calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        ''' compute linear activation'''
        return self.net_input(X)
    
    def predict(self, X):
        ''' return class label after unit step'''
        return np.where(self.activation(X) >= 0.0, 1, -1)



def plot_decision_regions(X, y, classifier, resolution = .02):
    '''decision edge's visualization'''
    '''wendy has not totally understand this part'''

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = .4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x = X[y == c1, 0], y = X[y == c1, 1], 
                    alpha = .8, c = cmap(idx), 
                    marker = markers[idx], label = c1)

def iris_example():
    '''get and scatter iris data'''
    df = pd.read_csv('iris.data', header = None)
    y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[:100, [0, 2]].values

    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()

    # plt.scatter(X[0:50, 0], X[0:50, 1], color = 'red', marker = 'o', label = 'setosa')
    # plt.scatter(X[51:100, 0], X[51:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
    # plt.xlabel('petal length (cm)')
    # plt.ylabel('sepal length (cm)')
    # plt.legend(loc = 'upper left')
    # plt.show()

    '''make classifier on the iris data'''
    # ppn = perceptron()
    # ppn.fit(X, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassifications')
    # plt.show()

    '''make 2-D dataset decision edge visulization'''
    # plot_decision_regions(X, y, classifier = ppn)
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('petal length [cm]')
    # plt.legend(loc = 'upper left')
    # plt.show()
    
    '''make plots of adalineGD between two different learning speed'''
    # fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

    # ada1 = adalineGD(n_iter = 10, eta = .01).fit(X, y)
    # ax[0].plot(range(1, len(ada1.cost_) + 1), 
    #             np.log10(ada1.cost_), marker = 'o')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('log(sum-squared-error)')
    # ax[0].set_title('Adaline - Learning rate .01')

    # ada2 = adalineGD(n_iter = 10, eta = .0001).fit(X, y)
    # ax[1].plot(range(1, len(ada2.cost_) + 1), 
    #             ada2.cost_, marker = 'o')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('Sum-squared-error')
    # ax[1].set_title('Adaline - Learning rate .0001')
    # plt.show()
    
    '''make plots again and deal the data to standard norm distribution first'''
    # ada = adalineGD(n_iter = 15, eta = .01)
    # ada.fit(X_std, y)
    # plot_decision_regions(X_std, y, classifier = ada)
    # plt.title('Adaline - Gradient Descent')
    # plt.xlabel('sepal length [standardized]')
    # plt.ylabel('petal length [standardized]')
    # plt.legend(loc = 'upper left')
    # plt.show()
    # plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sum - squared - error')
    # plt.show()

    ''''''
    ada = adalineSGD(n_iter = 15, eta = .01, random_state = 1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier = ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc = 'upper left')
    # plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    # plt.show()
    ada.partial_fit(X_std[0, :], y[0])



if __name__ == '__main__':
    iris_example()