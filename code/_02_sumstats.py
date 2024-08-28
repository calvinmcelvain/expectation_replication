doc = '''
- Calculates summary statistics for mean and individual forecasts
'''

import os
import pandas as pd
import numpy as np

# Working dir.
os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')

# Importing trim mean and ind. dfs
mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf_trim = pd.read_csv('cleaned_data/ind_spf_trim.csv')

# Setting df indexes
mean_spf_trim = mean_spf_trim.set_index('DATE')
ind_spf_trim = ind_spf_trim.set_index(['DATE', 'ID'])


###############################################
                ## SUM STATS ##
###############################################


def sumstats_df(df, end_date):
    '''
    Creates a new df for summary statistics.
     - Adds revision up, down, and no revision variables for each horizon (minus horizon 4)
    '''
    df = df[:end_date]
    df = df.drop(['CPI1', 'CPI2', 'CPI3', 'CPI4', 'CPI5', 'CPI6', 'rec'], axis=1)
    df = df * 100  # For easier interpretation 
    for n in range(0,4):
        df[f'ru_t{n}'] = (df[f'r_t{n}'] > 0).astype(int) # Revised up
        df[f'rd_t{n}'] = (df[f'r_t{n}'] < 0).astype(int) # Revised down
        df[f'rn_t{n}'] = (df[f'r_t{n}'] == 0).astype(int) # Did not revise
    return df


date = '2022-12-31' # Set desired date for summary statistics
ind_stats = sumstats_df(ind_spf_trim, date)
mean_stats = sumstats_df(mean_spf_trim, date)

# print(f"Number of Unique Forecasters: {ind_spf_trim[:date].index.get_level_values('ID').nunique()}")
# print(f"Average Forecaster per Forecaster: {np.mean(ind_spf_trim[:date].groupby(level='ID').size())}")
# ind_stats.describe().round(2)
# mean_stats.describe().round(2)