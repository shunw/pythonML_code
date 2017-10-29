import pandas as pd

df_adult = pd.read_csv('adult.data', header = None)
df_adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']

# print (df_adult.iloc[27, 1].strip() == '?')
print (df_adult['workclass'].where(df_adult['workclass'].strip() == '?'))