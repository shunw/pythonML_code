import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

import pip

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics  import roc_auc_score, average_precision_score

def check_qm(df, df_col_ls):
    '''
    CHECK the question mark number
    ==============================
    df_col_ls: list/ col name list
    df: dataframe to be checked
    '''
    for n in df_col_ls:
        if df[n].dtype == 'int64': continue
        
        qestmark_qty = sum([i for i in list(df[n].str.find("?")) if i != -1])
        if qestmark_qty == 0: continue
        print ('column name is {name}'.format(name = n))
        print ('question mark qty is {qty}'.format(qty = qestmark_qty))

def replace_qm(df, df_col_ls):
    '''
    REPLACE the question mark with np.nan
    =====================================
    df_col_ls: list/ col name list
    df: dataframe to be used to replace the question mark
    '''        
    for n in df_col_ls:
        if df[n].dtype == 'int64': continue
        df[n] = df[n].str.strip()
        df[n] = df[n].replace('?', 'N/A')
    return df

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

if __name__ == '__main__':
    df_adult = pd.read_csv('adult.data', header = None)
    name_col = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    df_adult.columns = name_col
    # print (df_adult.iloc[26:27, ])
    # check_qm(df_adult, name_col)
    
    
    '''
    DATA PREPROCESS
    ====================
    1. strip the str; replace the '?' with 'N/A'
    2. make the class_map, and set the data with 'N/A' label as the level -1
    3. replace the str/category into number. 
    '''
    df_adult = replace_qm(df_adult, name_col)
    # print (df_adult.shape)

    # drop all the "N/A" data
    for i in name_col: 
        # print (i)
        df_adult = df_adult[~df_adult[i].isin(["N/A"])]
        
    # print (df_adult.shape)

    cls_map_dict = make_class_map(df_adult, name_col)
    df_adult = do_class_map(df_adult, name_col, cls_map_dict)
    
    '''
    DATA DEAL
    ====================
    0. make the pair plot to check feature relationship
    1. make X, y; split into training data and test data; standarize the data; remove the related items. 
    2. choose the algorithm 
    3. verification/ plot the decision boundary

    addition: 
    may choose some algorithm to check the important feature
    reduce the data dimension
    '''

    
    # 0
    # according to the understanding of the col, [education <-> education num], [race <-> native country]

    # sns.set(style='whitegrid', context='notebook')    
    # g = sns.PairGrid(df_adult, vars =  name_col[:-1], hue = name_col[-1])
    # g = g.map_diag(plt.hist, histtype = 'step', linewidth = 3)
    # g = g.map_offdiag(plt.scatter)
    # g = g.add_legend()
    # plt.savefig('adult_pair.png')
    

    # 1 w/o moving the related items
    X = df_adult[name_col[:-1]].values
    y = df_adult[name_col[-1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    # # 2 --- algorithm lg
    # lr = LogisticRegression(C = 1000, random_state = 0)
    # lr.fit(X_train_std, y_train)
    # y_pred = lr.predict(X_test_std)
    # # print (y_pred)
    # y_prob = lr.predict_proba(X_test_std)[:, 1]


    # print('Precision: %.3f' % precision_score(y_true = y_test, y_pred = y_pred))
    # print('Recall: %.3f' % recall_score(y_true = y_test, y_pred = y_pred))
    # print('F1: %.3f' % f1_score(y_true = y_test, y_pred = y_pred))
    # print('roc_auc: %.3f' % roc_auc_score(y_true = y_test, y_score = y_prob))
    # print('avg_precision: %.3f' % average_precision_score(y_true = y_test, y_score = y_prob))

    # # 2-1 --- tunning parameter
    # # metric of the GridSearchCV: accuracy/ average_precision/ f1/ f1_micro/ f1_macro/ f1_weighted/ f1_samples/ neg_log_loss/ precision/ recall/ roc_auc

    # # lr, change the parameter of C, penalty, or change the metric of the scoring won't change the training score/ test score a lot. 
    # lr = LogisticRegression(penalty = 'l2', random_state = 0)
    # param_range = [10, 100, 1000, 10000, 100000]

    # # gs = GridSearchCV(estimator = lr, param_grid = {'C': param_range}, scoring = 'average_precision', cv = 10)
    # # gs = gs.fit(X_train_std, y_train)
    # # print (gs.best_score_)
    # # print (gs.best_params_)

    # train_scores, test_scores = validation_curve(estimator = lr, X = X_train_std, y = y_train, param_name = 'C', param_range = param_range, cv = 5, scoring = "roc_auc")
    # train_mean = np.mean(train_scores, axis = 1)
    # train_std = np.std(train_scores, axis = 1)

    # test_mean = np.mean(test_scores, axis = 1)
    # test_std = np.std(test_scores, axis = 1)
    
    # print ('train_mean: ', train_mean)
    # print ('train_std: ', train_std)
    # print ('test_mean: ', test_mean)
    # print ('test_std: ', test_std)

    # plt.plot(param_range, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
    # plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = .15, color = 'blue')

    # plt.plot(param_range, test_mean, color = 'green', marker = 's', markersize = 5, linestyle = '--', label = 'validation accuracy')
    # plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha = .15, color = 'green')
    
    # plt.grid()
    # plt.xscale('log')
    # plt.legend(loc = 'lower right')
    # plt.xlabel('Parameter C')
    # plt.ylabel('Accuracy')
    # plt.ylim([.5, .7])
    # plt.show()
    
    # 3 decision tree
    tree = DecisionTreeClassifier(random_state = 0)
    param_range = ['entropy', 'gini' ]
    depth_range = [3, 5, 7, 8, 9, 10, 11]
    # tree.fit(X_train, y_train)

    # gs = GridSearchCV(estimator = tree, param_grid = {'criterion': param_range, 'max_depth': depth_range}, scoring = 'accuracy', cv = 5)
    # gs = gs.fit(X_train, y_train)
    # print (gs.best_score_)
    # print (gs.best_params_)

    tree_1 = DecisionTreeClassifier(criterion = "gini", random_state = 0)
    train_scores, test_scores = validation_curve(estimator = tree_1, X = X_train, y = y_train, param_name = 'max_depth', param_range = depth_range, cv = 5, scoring = "roc_auc")
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    
    print ('train_mean: ', train_mean)
    print ('train_std: ', train_std)
    print ('test_mean: ', test_mean)
    print ('test_std: ', test_std)

    plt.plot(depth_range, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
    plt.fill_between(depth_range, train_mean + train_std, train_mean - train_std, alpha = .15, color = 'blue')

    plt.plot(depth_range, test_mean, color = 'green', marker = 's', markersize = 5, linestyle = '--', label = 'validation accuracy')
    plt.fill_between(depth_range, test_mean + test_std, test_mean - test_std, alpha = .15, color = 'green')
    
    plt.grid()
    plt.xscale('log')
    plt.legend(loc = 'lower right')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    # plt.ylim([.5, .7])
    plt.show()


    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    '''
    QUESTION: 
    1. for the pair plot
        - if the features are too much, the pair plot will be created very slow (like adult, it has like 14 features), even if you want to change the scale of the plot, it will take a long time. 
    2. for the learning curve
        - run one learning curve and it's hard to see the difference between the learning score between different parameter
    
    NEXT: 
        - tune the parameter/ 
            - [NA] check if there is any other parameter need to be tuned in the lr/
            - [NA] begin to tune with the SVM
            - ! according to some paper, try decision tree. And find some changes when tune the parameter. 

        - check the learning curves/ validation curves
            - 
        - ? how to decrease the dimension. 
            - after decrease the dimension, is it possible to run the pair plot
        - remove all the missing data and check if the score will be changed. / also need to check the data qty before and after removing the missing data
    '''