import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics  import roc_auc_score, average_precision_score

from sklearn.tree import export_graphviz

def check_qm(df): 
    for n in list(df.columns): 
        # print (n)
        qestmark_qty = sum([1 for i in list(df[n].str.find("?")) if i != -1])
        
        if qestmark_qty == 0: continue
        print ('column name is {name}'.format(name = n))
        print ('question mark qty is {qty}'.format(qty = qestmark_qty))

def make_class_map(df, df_col_ls):
    '''
    MAKE class map for each str columns
    =====================================
    df_col_ls: list/ col name list
    df: dataframe to be used to replae the question mark
    cls_map_dict: dict/ connect column name with the class mapping
    '''
    cls_map_dict = dict()
    for n in df_col_ls:
        if df[n].dtype == 'int64': continue
        # print (df[n])
        # print (np.unique(df[n]))
        temp_dict = dict()
        for idx,label in enumerate(np.unique(df[n])): 
            if label != 'N/A': temp_dict[label] = idx
            else: temp_dict[label] = -1
        cls_map_dict[n] = temp_dict

        # print (cls_map_dict)
    return cls_map_dict

def do_class_map(df, df_col_ls, cls_map_dict):
    '''
    MAP the category into int with class map
    '''
    for n in df_col_ls:
        if df[n].dtype == 'int64': continue
        df[n] = df[n].map(cls_map_dict[n])
    return df

class mushroom_ana:
    def __init__(self, df): 
        self.raw_data = df
        self.col_names = self.raw_data.columns

    def _dp_remove_missing(self):
        '''
        input: dataframe/ rawdata
        output: 
            1. deal with missing data
            2. make the data split
        '''
        check_qm(self.raw_data)

    def _dp_data_2split(self):
        # turn the data into int/float
        # split the data
        cls_map_dict = make_class_map(self.raw_data, self.col_names)
        self.raw_data = do_class_map(self.raw_data, self.col_names, cls_map_dict)
        # print (cls_map_dict)

        self.y = self.raw_data[self.col_names[0]].values
        self.X = self.raw_data[self.col_names[1:]].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = .3, random_state = 0)

    def _paratune(self, alg, param_grid, score_name):
        gs = GridSearchCV(estimator = alg, param_grid = param_grid, scoring = score_name, cv = 5)
        gs = gs.fit(self.X, self.y)
        # print (gs.best_score_)
        # print (gs.best_params_)

    def _learning_cur_plot(self, alg, param_name, param_range, score_name):
        train_scores, test_scores = validation_curve(estimator = alg, X = self.X_train, y = self.y_train, param_name = param_name, param_range = param_range, cv = 5, scoring = score_name)
        train_mean = np.mean(train_scores, axis = 1)
        train_std = np.std(train_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        
        # print ('train_mean: ', train_mean)
        # print ('train_std: ', train_std)
        # print ('test_mean: ', test_mean)
        # print ('test_std: ', test_std)

        plt.plot(param_range, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training {score_name}'.format(score_name = score_name))
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = .15, color = 'blue')

        plt.plot(param_range, test_mean, color = 'green', marker = 's', markersize = 5, linestyle = '--', label = 'validation {score_name}'.format(score_name = score_name))
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha = .15, color = 'green')
        
        plt.grid()
        plt.xscale('log')
        plt.legend(loc = 'lower right')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        # plt.ylim([.5, .7])
        plt.show()        

    def decision_tree(self):
        tree = DecisionTreeClassifier(random_state = 0)
        param_range = ['entropy', 'gini' ]
        depth_range = [3, 5, 7, 8, 9, 10, 11]
        param_grid = {'criterion': param_range, 'max_depth': depth_range}

        '''tune paramter'''
        self._paratune(tree, param_grid, 'accuracy')

        '''plot the learning curve'''
        tree = DecisionTreeClassifier(random_state = 0, criterion = 'entropy')
        self._learning_cur_plot(tree, 'max_depth', depth_range, "roc_auc")

        tree = DecisionTreeClassifier(criterion = "entropy", random_state = 0, max_depth = 7)
        tree.fit(self.X_train, self.y_train)
        self.y_pred = tree.predict(self.X_test)
        self.y_prob = tree.predict_proba(self.X_test)[:, 1]

        # export_graphviz(tree, out_file = 'tree.dot', feature_names = self.col_names[1:])

    def preci_scores(self):
        return (precision_score(y_true = self.y_test, y_pred = self.y_pred))

    def accuracy_scores(self):
        return (accuracy_score(y_true = self.y_test, y_pred = self.y_pred))

    def roc_auc_scores(self):
        return (roc_auc_score(y_true = self.y_test, y_score = self.y_prob))
    
    def test(self):
        print (self.raw_data.shape)

if __name__ == '__main__':
    df_mushroom = pd.read_csv('agaricus-lepiota.data', header = None)
    name_col = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    df_mushroom.columns = name_col

    m_ana = mushroom_ana(df_mushroom)
    m_ana._dp_data_2split()
    m_ana.decision_tree()
    print ('precision score is: {precision_score: .3f}'.format(precision_score = m_ana.preci_scores()))
    
    print ('accuracy score is: {accuracy_score: .3f}'.format(accuracy_score = m_ana.accuracy_scores()))

    print ('score is:{roc_auc_score: .3f}'.format(roc_auc_score = m_ana.roc_auc_scores()))

    '''
    compare version:
        - try the data with decision tree with/ without dealing with the missing data
        - compare the accuracy

    step 1: make a clear data. 
        - deal with the missing data
        - turn the string into number if needed / only "stalk-root" has the question mark
        - make the data split [done]

    step 2: check all the algorithm
        - use the algorithm
        - tune the parameter [decison tree done/ ]
        - check the learning curve [decision tree done/ ]
        - ! check the accuracy [decision tree done/ ]
        - ! try to find the important parameter 
            * lr
            * svm
            * decision tree (does this need to turn the data from string into int? )
            * naive bayes (any other naive bayes could be used except the gaussion NB?)
    
    
    '''