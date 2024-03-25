import pandas as pd
import numpy as np

mean_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.csv')
ind_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/ind_spf_trim.csv')

mean_spf = mean_spf.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])

mean_stats = pd.DataFrame()
ind_stats = pd.DataFrame()

###############################################
                ## STATS ##
###############################################
mean_spf = mean_spf[:'2016-12-31']  # Filter data


def sum_stats(df, v):
    for j in range(4):
        v.loc[j, 'mean_revisions'] = np.mean(df[f'r_t{j}'] * 100)
        v.loc[j, 'std_revisions'] = np.std(df[f'r_t{j}'] * 100)
        upward = 0
        downward = 0
        na = 0
        total = 0
        for i in df.index:
            if pd.notna(df.loc[i, f'r_t{j}']):
                if df.loc[i, f'r_t{j}'] > 0:
                    upward += 1
                    total += 1
                elif df.loc[i, f'r_t{j}'] < 0:
                    downward += 1
                    total += 1
                else:
                    na += 1
                    total += 1
                v.loc[j, 'revisions_up'] = upward/total
                v.loc[j, 'revisions_downn'] = downward/total
                v.loc[j, 'revisions_na'] = na/total
    
    for j in range(5):
        v.loc[j, 'mean_errors'] = np.mean(df[f'e_t{j+1}'] * 100)
        v.loc[j, 'std_errors'] = np.std(df[f'e_t{j+1}'] * 100)

sum_stats(mean_spf, mean_stats)
sum_stats(ind_spf, ind_stats)

