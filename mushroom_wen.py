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
from sklearn.metrics  import roc_auc_score, average_precision_score

def check_qm(df): 
    for n in list(df.columns): 
        # print (n)
        qestmark_qty = sum([1 for i in list(df[n].str.find("?")) if i != -1])
        
        if qestmark_qty == 0: continue
        print ('column name is {name}'.format(name = n))
        print ('question mark qty is {qty}'.format(qty = qestmark_qty))


class mushroom_ana:
    def __init__(self, df): 
        self.raw_data = df

    def data_prepro(self):
        '''
        input: dataframe/ rawdata
        output: 
            1. deal with missing data
            2. make the data split
        '''
        check_qm(self.raw_data)

    def test(self):
        print (self.raw_data.shape)

if __name__ == '__main__':
    df_mushroom = pd.read_csv('agaricus-lepiota.data', header = None)
    name_col = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    df_mushroom.columns = name_col

    m_ana = mushroom_ana(df_mushroom)
    m_ana.data_prepro()
    
    
    '''
    compare version:
        - try the data with decision tree with/ without dealing with the missing data
        - compare the accuracy

    step 1: make a clear data. 
        - deal with the missing data
        - turn the string into number if needed / only "stalk-root" has the question mark
        - make the data split

    step 2: check all the algorithm
        - use the algorithm
        - tune the parameter
        - check the learning curve
        - ! check the accuracy
        - ! try to find the important parameter 
            * lr
            * svm
            * decision tree (does this need to turn the data from string into int? )
            * naive bayes (any other naive bayes could be used except the gaussion NB?)
    
    
    '''