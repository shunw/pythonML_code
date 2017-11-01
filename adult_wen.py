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
        df[n] = df[n].replace(' ?', np.NaN)
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
        cls_map_dict[n] = {label:idx for idx,label in enumerate(np.unique(df[n]))}
    return cls_map_dict



if __name__ == '__main__':
    df_adult = pd.read_csv('adult.data', header = None)
    name_col = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    df_adult.columns = name_col
    # print (df_adult.iloc[26:27, ])
    # check_qm(df_adult, name_col)
    
    df_adult = replace_qm(df_adult, name_col)
    a = make_class_map(df_adult, name_col)
    print (a)