import pandas as pd
import numpy as np
from _001cleaning import filter_date

###############################################
                ## REVISIONS ##
###############################################

#####################
    ## MEAN SPF ##
#####################

mean_spf_trim = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.cvs')
vintage_trim = pd.read_csv('Documents/thesis/_replication/cleaned_data/vintage_trim.cvs')

working_mean_data = pd.merge(mean_spf_trim, vintage_trim[['t', 't0', 't1', 't1', 't2', 't3', 't4']], on='yrq', how='right')
s = pd.read_stata('Documents/thesis/models/cg/Replication_files/workfiles/us_data_updated.dta')
l = pd.read_stata('Documents/thesis/models/cg/Replication_files/Data/Macro-data---FRED.dta')
j = pd.read_stata('Documents/thesis/models/diagnostic/programs/data/spf/iCPI.dta')

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

revision_means, revision_mean = mean_revisions(working_mean_data)
revision_stds, revision_std = std_revisions(working_mean_data)

#####################
## INDIVIDUAL SPF ##
#####################

ind_spf_trim = pd.read_csv('Documents/thesis/_replication/cleaned_data/ind_spf_trim.cvs')
working_ind_data = pd.merge(ind_spf_trim, vintage_trim[['t', 't0', 't1', 't1', 't2', 't3', 't4']], on='yrq', how='right')

# Filter date
working_ind_data = filter_date(working_ind_data, '1981Q3', 0)



###############################################
                ## Errors ##
###############################################