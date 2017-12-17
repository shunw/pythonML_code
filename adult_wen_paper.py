import adult_wen

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


class alg_accuracy_change:
    def __init__(self, raw_data, train_qty_ls):
        '''
        we just compare the decision tree and naive bayes. so there are only these two alg here.

        raw_data: is all the data, with panda's format
        train_qty_ls: is the list, to list all the instances qty for the train data. they are different to see the change of the accuracy with the increase of the instance. 

        '''
        self.raw_data = raw_data
        self.train_qty_ls = train_qty_ls
        self._Xy_data()
    
    def _Xy_data(self):
        self.ncol = self.raw_data.shape[1]
        
        self.X = self.raw_data[self.raw_data.columns[:self.ncol-1]].values
        self.y = self.raw_data[self.raw_data.columns[-1]].values
        
    def NB_acc(self, train_size):
        run_q = 20
        acc = np.zeros(run_q)
   
        gnb = GaussianNB()
        
        for r in range(run_q):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size = train_size, random_state = r)
            gnb.fit(X_train, y_train)
            y_pred = gnb.predict(X_test)
            acc[r] = accuracy_score(y_true = y_test, y_pred = y_pred)
        # output is the list, [mean - 2*std, mean, mean + 2*std]
        
        return [acc.mean() - 2 * acc.std(), acc.mean(), acc.mean() + 2 * acc.std()]

    def DTree_acc(self, train_size): 
        run_q = 20
        acc = np.zeros(run_q)

        dtree = DecisionTreeClassifier(criterion = "gini", random_state = 0, max_depth = 7)

        for r in range(run_q): 
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size = train_size, random_state = r)
            dtree.fit(X_train, y_train)
            y_pred = dtree.predict(X_test)
            acc[r] = accuracy_score(y_true = y_test, y_pred = y_pred)

        # output is the list, [mean - 2*std, mean, mean + 2*std]
        
        return [acc.mean() - 2 * acc.std(), acc.mean(), acc.mean() + 2 * acc.std()]

    def try_out(self):
        a = self.NB_acc(500)
        print (a)



if __name__ == '__main__':
    df_adult = adult_wen.adult_data_preprocess()

    train_qty_ls = range(2500, 30000, 2500)
    acc = alg_accuracy_change(df_adult, train_qty_ls)
    
    nb_plot_data = dict()
    dt_plot_data = dict()
    for q in train_qty_ls: 
        nb_plot_data[q, q, q] = acc.NB_acc(q)
        dt_plot_data[q, q, q] = acc.DTree_acc(q)

    
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    for k in nb_plot_data.keys(): 
        ax1.plot(list(k), nb_plot_data[k], marker = '_', markersize = 14, color = 'cyan')
        ax1.plot(list(k), dt_plot_data[k], marker = '_', markersize = 14, color = 'magenta')
        # print (list(k))
        # print (v)


    plt.show()

    '''
    Decision tree's learning curve is a little like the paper
    Naive Bayes's learning cuve is not much like the paper. =<
    Data: adult
    Paper link: http://robotics.stanford.edu/~ronnyk/nbtree.pdf
    '''



    
