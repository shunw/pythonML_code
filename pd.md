resample document:
How to make the time series continuouly.
https://stackoverflow.com/questions/17001389/pandas-resample-documentation

remove the non-ascii 
replace the non-ascii with '-'
df.replace({r'[^\x00-\x7f]+':'-'}, regex=True, inplace=True)

split the string in one column and split to two columns
s = pd.DataFrame(df_manager.Seasons.str.split('-').tolist(), columns = ['Season_start','Season_end_0'])
    df_manager = df_manager.join(s)

df_manager['Season_start'] has uncontinuous year, w the following method, make the column with continuous year. 
df_manager_2 = df_manager_2.set_index('Season_start').resample('AS').asfreq().fillna(0)
    df_manager_2 = df_manager_2.reset_index()


remove the NaT value in the dataframe
df_score = df_score[pd.notnull(df_score['Season_'])]