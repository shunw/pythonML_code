import numpy as np
import pandas as pd
import re

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
    2. make the class_map
    3. replace the str/category into number. 
    '''
    df_adult = replace_qm(df_adult, name_col)
    cls_map_dict = make_class_map(df_adult, name_col)
    df_adult = do_class_map(df_adult, name_col, cls_map_dict)
    


    '''
    QUESTION: 
    1. for the ? item, how to deal with it? 
        - replace it with np.nan at the very beginning? (this would make the np.unique formula failure, and hard to transfer the string data into number, which is necessary during the data preprocess)
        - 
    '''