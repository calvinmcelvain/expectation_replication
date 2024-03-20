import pandas as pd
import numpy as np
from _001cleaning import filter_date

###############################################
                ## Revisions ##
###############################################

mean_spf_trim = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.cvs')
vintage_trim = pd.read_csv('Documents/thesis/_replication/cleaned_data/vintage_trim.cvs')

mean_spf_trim.set_index('Unnamed: 0', inplace=True)
vintage_trim.set_index('yrq', inplace=True)
working_data = pd.merge(mean_spf_trim, vintage_trim[['t', 't0', 't1', 't1', 't2', 't3', 't4']], on='yrq', how='right')
working_data = pd.merge(working_data, ind_spf_trim[['']], on='yrq', how='right')
s = pd.read_stata('Documents/thesis/models/cg/Replication_files/workfiles/us_data_updated.dta')
l = pd.read_stata('Documents/thesis/models/cg/Replication_files/Data/Macro-data---FRED.dta')
j = pd.read_stata('Documents/thesis/models/diagnostic/programs/data/vintage/CPI.dta')

# Filter date
working_data = filter_date(working_data, '1981Q3', 0)

# Calculating revisions
def revisions(dataframe, a, b):
    for n in range(a, b):
        for i in range(1, 5):
            dataframe.loc[n, f'r_t{i}'] = dataframe.loc[n + 1, f'f_t{i}'] - dataframe.loc[n, f'f_t{i + 1}']

revisions(working_data, 64, 204)

# Calculating mean and std of revisions (all horizons)
def mean_revisions(dataframe):
    list = []
    for i in range(1, 5):
        list.append(np.mean(dataframe[f'r_t{i}']))
    a = np.array(list)
    b = np.mean(a)
    return a, b

def std_revisions(dataframe):
    list = []
    stacked_df = pd.concat([dataframe[f'r_t{k}'] for k in range(1, 5)], axis=1).stack()
    for i in range(1, 5):
        list.append(np.std(dataframe[f'r_t{i}']))
    a = np.array(list)
    b = np.std(stacked_df)
    return a, b

revision_means, revision_mean = mean_revisions(working_data)
revision_stds, revision_std = std_revisions(working_data)

###############################################
                ## Errors ##
###############################################

