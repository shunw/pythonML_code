# -*- coding: utf-8 -*-
'''
Compressing Data via Dimensionality Reduction: 
1. PCA - principal component analysis （无监督）
    无监督线性数据转换技术，帮助我们基于特征之间的关系识别出数据内在的模式 - 高维数据中找到最大方差的方向，并将数据映射到一个维度不大于原始数据的新的子空间上
    流程：
    1. 对原始d维数据集做标准化处理
    2. 构造样本的协方差矩阵
    3. 计算协方差矩阵的特征值和相应的特征向量
    4. 选择与前k个最大特征值对应的特征向量，其中k为新特征空间的维度
    5. 通过前k个特征向量构建映射矩阵W
    6. 通过映射矩阵W将d维的输入数据集X转换到新的k维特征子空间。
2. LDA - linear discriminant analysis （监督）
    线性判别分析，可以提高分析过程中的计算效率。同时，对于不适用于正则化的模型。
    目标是发现可以最优化分类的特征子空间。
    假设LDA的数据呈正态分布，且样本的特征从统计上来讲是相互独立的。不过即使一个或多个假设没有满足，LDA仍可以很好的完成降维。
    流程：
    1. 对d维数据集进行标准化处理（d为特征的数量）
    2. 对每一类别，计算d维的均值向量
    3. 构造类间的散布矩阵SB 以及类内的散布矩阵SW
    4. 计算矩阵SW^(-1)SB的特征值及对应的特征向量
    5. 选取前k个特征值所对应的特征向量，构造一个dxk维的转换矩阵W，其中特征向量以列的形式排列
    6. 使用转换矩阵W将样本映射到新的特征子空间上
3. kernel principal component analysis

'''
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, resolution = .02):

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
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = .8, c = cmap(idx), marker = markers[idx], label = cl)

class wine_data:
    def __init__(self):
        df_wine = pd.read_csv('wine.data', header = None)
        self.X, self.y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = .3, random_state = 0)
        self.sc = StandardScaler()

        self.X_train_std = self.sc.fit_transform(self.X_train)
        self.X_test_std = self.sc.transform(self.X_test)

    def train_data(self):
        return self.X_train_std, self.y_train
        
    def test_data(self):
        return self.X_test_std, self.y_test

    def all_data(self):
        return self.X, self.y

def PCA_related():
    data = wine_data()
    X_train_std, y_train = data.train_data()
    X_test_std, y_test = data.test_data()

    '''get the eigenvalus'''
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # print ('\nEigenvalues \n{eigen_vals}'.format(eigen_vals = eigen_vals))
    # print eigen_vecs

    '''plot variance explained ratios'''
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]
    cum_var_exp = np.cumsum(var_exp)

    # plt.bar(range(1, 14), var_exp, alpha = .5, align = 'center', label = 'individual explained variance')
    # plt.step(range(1, 14), cum_var_exp, where = 'mid', label = 'cumulative explained variance')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal components')
    # plt.legend(loc = 'best')
    # plt.show()

    '''feature transformation'''
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse = True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    # print ('Matrix W:\n', w)

    '''transform 13d to 2d'''
    # print X_train_std[0].dot(w)
    X_train_pca = X_train_std.dot(w)

    '''plot first and second main feature plot'''
    # colors = ['r', 'b', 'g']
    # markers = ['s', 'x', 'o']
    # for l, c, m in zip(np.unique(y_train), colors, markers):
    #     plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c = c, label = l, marker = m)
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.legend(loc = 'lower left')
    # plt.show()

    '''PCA in the scikit-learn'''
    # pca = PCA(n_components = 2)
    # lr = LogisticRegression()
    # X_train_pca = pca.fit_transform(X_train_std)
    # X_test_pca = pca.transform(X_test_std)
    # lr.fit(X_train_pca, y_train)
    # plot_decision_regions(X_train_pca, y_train, classifier = lr)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.legend(loc = 'lower left')
    # plt.show()

    '''check in the test data'''
    # plot_decision_regions(X_test_pca, y_test, classifier = lr)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.legend(loc = 'lower left')
    # plt.show()

    '''check PCA all principal components by the explained variation ratio'''
    pca = PCA(n_components = None)
    X_train_pca = pca.fit_transform(X_train_std)
    # print pca.explained_variance_ratio_

def LDA_related():
    data = wine_data()
    X_train_std, y_train = data.train_data()
    X_test_std, y_test = data.test_data()
    X, y = data.all_data()

    np.set_printoptions(precision = 4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis = 0))
        # print ('MV {label}: {value} \n').format(label = label, value = mean_vecs[label - 1])

    '''calculate the within class scatter matrix'''
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X[y == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
            # print class_scatter
        S_W += class_scatter
    # print ('Within-class scatter matrix: {shape0}x{shape1}'.format(shape0 = S_W.shape[0], shape1 = S_W.shape[1]))

    # print ('Class label distribution: {label}'.format(label = np.bincount(y_train)[1:]))

    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter
    # print ('Scaled within-class scatter matrix: {shape0}x{shape1}'.format(shape0 = S_W.shape[0], shape1= S_W.shape[1]))

    '''compute the between-class scatter matrix SB'''
    mean_overall = np.mean(X_train_std, axis = 0)
    d = 13
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    # print ('Between-lass scatter matrix: {shape0}x{shape1}'.format(shape0 = S_B.shape[0], shape1 = S_B.shape[1]))

    '''select linear discriminants'''
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    # print eigen_vecs
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True)
    # print ('Eigenvalues in decreasing order:\n')
    # for eigen_val in eigen_pairs:
    #     print (eigen_val[0])

    # tot = sum(eigen_vals.real)
    # discr = [(i / tot) for i in sorted(eigen_vals.real, reverse = True)] 
    # cum_discr = np.cumsum(discr)
    # plt.bar(range(1, 14), discr, alpha = .5, align = 'center', label = 'individual "discriminability"')
    # plt.step(range(1, 14), cum_discr, where = 'mid', label = 'cumulative "discriminability"')
    # plt.ylabel('"discriminability" ratio')
    # plt.ylim([-.1, 1.1])
    # plt.legend(loc = 'best')
    # plt.show()

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
    # print('Matrix W: \n', w)

    # print eigen_pairs[0][1][:, np.newaxis].real

    '''projecting samples onto the new feature space'''
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    # for l, c, m in zip(np.unique(y_train), colors, markers):
    #     plt.scatter(X_train_lda[y_train == l, 0] * (-1),
    #                 X_train_lda[y_train == l, 1] * (-1),
    #                 c = c, label = l, marker = m
    #     )
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.legend(loc = 'lower right')
    # plt.show()

    '''lda in the scikit-learn'''
    lda = LDA(n_components = 2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    # on the training data
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    # plot_decision_regions(X_train_lda, y_train, classifier = lr)
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.legend(loc = 'lower left')
    # plt.show()

    # on the test data
    X_test_lda = lda.transform(X_test_std)
    # plot_decision_regions(X_test_lda, y_test, classifier = lr)
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.legend(loc = 'lower left')
    # plt.show()

def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation.

    Parameters
    -------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel
    
    n_components: int
        Number of principal components to return

    Returns
    -------------
    X_pc: {NumPy ndarray}, shape = [n_sampels, k_features] 
        Projected dataset
    lambdas: list
        Eigenvalues
    '''
    # Calculate pairwise squared Euclidean distances in the MxN dimensional dataset. 
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix. 
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))
    
    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)


def kernel_pca():
    
    '''example 2'''
    '''data making'''
    X, y = make_circles(n_samples = 1000, random_state = 123, noise = .1, factor = .2)
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    # plt.show()

    '''deal with the standard PCA'''
    scikit_pca = PCA(n_components = 2)
    X_spca = scikit_pca.fit_transform(X)
    # fit, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
    # ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    # ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    # ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + .02, color = 'red', marker = '^', alpha = .5)
    # ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - .02, color = 'blue', marker = 'o', alpha = .5)
    # ax[0].set_xlabel('PC1')
    # ax[0].set_ylabel('PC2')
    # ax[1].set_ylim([-1, 1])
    # ax[1].set_yticks([])
    # ax[1].set_xlabel('PC1')
    # plt.show()

    '''deal with the RBF pca'''
    X_kpca, lambdas = rbf_kernel_pca(X, gamma = 15, n_components = 2)
    # fit, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
    # ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    # ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    # ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + .02, color = 'red', marker = '^', alpha = .5)
    # ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - .02, color = 'blue', marker = 'o', alpha = .5)
    # ax[0].set_xlabel('PC1')
    # ax[0].set_ylabel('PC2')
    # ax[1].set_ylim([-1, 1])
    # ax[1].set_yticks([])
    # ax[1].set_xlabel('PC1')
    # plt.show()

    '''example 1'''
    ''' make non-linear data'''
    X, y = make_moons(n_samples = 100, random_state = 123)
    # plt.scatter(X[y == 0, 0], X[y==0, 1], color = 'red', marker = '^', alpha = .5)
    # plt.scatter(X[y == 1, 0], X[y==1, 1], color = 'blue', marker = 'o', alpha = .5)
    # plt.show()

    ''' try the non-linear data on the standard PCA first'''
    scikit_pca = PCA(n_components = 2)
    X_spca = scikit_pca.fit_transform(X)
    # fit, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
    # ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    # ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    # ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + .02, color = 'red', marker = '^', alpha = .5)
    # ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - .02, color = 'blue', marker = 'o', alpha = .5)

    # ax[0].set_xlabel('PC1')
    # ax[0].set_ylabel('PC2')
    # ax[1].set_ylim([-1, 1])
    # ax[1].set_yticks([])
    # ax[1].set_xlabel('PC1')
    # plt.show()

    ''' try the rbf_kernel_pca '''
    X_kpca, lambdas = rbf_kernel_pca(X, gamma = 15, n_components = 2)
    # fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
    # ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    # ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    # ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + .02, color = 'red', marker = '^', alpha = .5)
    # ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - .02, color = 'blue', marker = 'o', alpha = .5)
    # ax[0].set_xlabel('PC1')
    # ax[0].set_ylabel('PC2')
    # ax[1].set_ylim([-1, 1])
    # ax[1].set_yticks([])
    # ax[1].set_xlabel('PC1')
    # ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    # ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    # plt.show()

    '''add new dataset, to use the rbf_kernel transform again'''
    alphas = X_kpca
    x_new = X[25]
    # print x_new
    # print alphas[25]
    x_reproj = project_x(x_new, X, gamma = 15, alphas = alphas, lambdas = lambdas)
    # print x_reproj

    '''use scikit-learn default kernel PCA'''
    scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
    X_skernpca = scikit_kpca.fit_transform(X)

    plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color = 'red', marker = '^', alpha = .5)
    plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color = 'blue', marker = 'o', alpha = .5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()



if __name__ == '__main__':
    kernel_pca()

