import pandas as pd
import numpy as np

###############################################
            ## GENERAL CLEANING ##
###############################################

#####################
    ## SPF DATA ##
#####################

### Mean Forecasts ###
mean_spf = pd.read_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv")
mean_spf.to_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git

mean_spf_trim = mean_spf
mean_spf_trim['yrq'] = mean_spf_trim['YEAR'].astype(str) + '_Q' + mean_spf_trim['QUARTER'].astype(str)
mean_spf_trim = mean_spf_trim.fillna("")
mean_spf_trim = mean_spf_trim.drop(['CPIA', 'CPIB', 'CPIC'], axis=1)
mean_spf_trim = mean_spf_trim.rename(columns={'CPI1':'f_t0', 'CPI2': 'f_t1', 'CPI3': 'f_t2', 'CPI4': 'f_t3', 'CPI5': 'f_t4', 'CPI6': 'f_t5'})

def filter_date(dataframe, a, b):
    if a != 0:
        c = dataframe[(dataframe['yrq'] >= a)]
        if b != 0:
            d = c[(c['yrq'] <= b)]
            return d
        else:
            return c
    else:
        if b != 0:
            d = c[(c['yrq'] <= b)]
            return d
        else:
            return c

mean_spf_trim = filter_date(mean_spf_trim, '1981_Q3', 0)
mean_spf_trim = mean_spf_trim.reset_index()

### Individual Forecasts ###
ind_spf = pd.read_csv("Documents/thesis/_replication/data/spf_ind_cpi.csv")
ind_spf.to_csv("Documents/thesis/_replication/data/spf_ind_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git

ind_spf_trim = ind_spf
ind_spf_trim['yrq'] = ind_spf_trim['YEAR'].astype(str) + '_Q' + ind_spf_trim['QUARTER'].astype(str)
ind_spf_trim = ind_spf_trim.drop(['CPIA', 'CPIB', 'CPIC'], axis=1)
ind_spf_trim = ind_spf_trim.rename(columns={'CPI1':'f_t0', 'CPI2': 'f_t1', 'CPI3': 'f_t2', 'CPI4': 'f_t3', 'CPI5': 'f_t4', 'CPI6': 'f_t5'})


ind_spf_trim = filter_date(ind_spf_trim,'1981_Q3', 0)
ind_spf_trim = ind_spf_trim.reset_index()

#####################
  ## VINTAGE DATA ##
#####################
vintage = pd.read_csv("Documents/thesis/_replication/data/vintage_cpi.csv")
vintage.to_csv("Documents/thesis/_replication/data/vintage_cpi.csv", sep=',', index=False)    # Original file had CRLF endings, changed to LF for Git

vintage_trim = vintage
vintage_trim['DATE'] = pd.to_datetime(vintage_trim['DATE'], format='%Y:%m')
vintage_trim.set_index('DATE', inplace=True)
vintage_trim = vintage_trim.resample('Q').mean()    # Data is collected monthly, mean aggregating to quarterly to match forecasts
vintage_trim.reset_index(inplace=True)
vintage_trim['yrq'] = vintage_trim['DATE'].dt.strftime('%Y_Q') + vintage_trim['DATE'].dt.quarter.astype(str)
vintage_trim['year'] = vintage_trim['yrq'].str.split('_').str[0].astype(int)
vintage_trim['quarter'] = vintage_trim['yrq'].str.split('Q').str[1].astype(int)
vintage_trim.drop(columns=['DATE'], inplace=True)

vintage_trim = filter_date(vintage_trim, '1965_Q1', 0)
vintage_trim = vintage_trim.reset_index()

def drop_cols(dfcols, a, b, di):
    columns = []
    for var in dfcols:
        split = var.split('I')
        if len(split) >= 2:
            for i in range(a, b):
                if split[1][0:2].startswith(f'{i}'):
                    columns.append(var)
    for e in di:
        columns.remove(e)
    return columns

vintage_trim = vintage_trim.drop(columns=drop_cols(vintage_trim.columns, 64, 95, ['CPI94Q3', 'CPI94Q4']))

###############################################
              ## CALCULATIONS ##
###############################################

### Vintage Growth ###
def cpi_growth(dataframe):
    for t in range(1, len(dataframe)-5):
        y = dataframe.loc[t - 1, 'year']
        q = dataframe.loc[t - 1, 'quarter']
        
        if q <= 3:
            if y <= 1994 and pd.notna(dataframe.loc[t - 1, 'CPI94Q3']):
                dataframe.loc[t, 't'] = dataframe.loc[t - 1, 'CPI94Q3']
            else:
                y = dataframe.loc[t, 'year']
                q = dataframe.loc[t, 'quarter']
                dataframe.loc[t, 't'] = dataframe.loc[t - 1, f'CPI{str(y)[2:4]}Q{q}']
        else:
            if y <= 1994 and pd.notna(dataframe.loc[t - 1, 'CPI94Q3']):
                dataframe.loc[t, 't'] = dataframe.loc[t - 1, 'CPI94Q3']
            else:
                y = dataframe.loc[t, 'year']
                q = dataframe.loc[t, 'quarter']
                dataframe.loc[t, 't'] = dataframe.loc[t - 1, f'CPI{str(y)[2:4]}Q{q}']
            
        for i in range(0, 5):
            y_i = dataframe.loc[t + i + 1, 'year']
            q_i = dataframe.loc[t + i + 1, 'quarter']
            if q_i <= 3:
                if y_i <= 1994 and pd.notna(dataframe.loc[t + i, 'CPI94Q3']):
                    dataframe.loc[t, f't{i}'] = ((dataframe.loc[t + i, 'CPI94Q3'] / dataframe.loc[t, 't']) - 1) * 100
                else:
                    dataframe.loc[t, f't{i}'] = ((dataframe.loc[t + i, f'CPI{str(y_i)[2:4]}Q{str(q_i)}'] / dataframe.loc[t, 't']) - 1) * 100
            else:
                if y_i <= 1994 and pd.notna(dataframe.loc[t + i, 'CPI94Q3']):
                    dataframe.loc[t, f't{i}'] = ((dataframe.loc[t + i, 'CPI94Q3'] / dataframe.loc[t, 't']) - 1) * 100
                else:
                    dataframe.loc[t, f't{i}'] = ((dataframe.loc[t + i, f'CPI{str(y_i)[2:4]}Q{str(q_i)}'] / dataframe.loc[t, 't']) - 1) * 100

cpi_growth(vintage_trim)

### Revisions ###

# Mean SPF data
def revisions(dataframe):
    for n in range(0, len(dataframe)-2):
        for i in range(1, 5):
            dataframe.loc[n, f'r_t{i}'] = dataframe.loc[n + 1, f'f_t{i}'] - dataframe.loc[n, f'f_t{i + 1}']

revisions(mean_spf_trim)

# Individual SPF data
def revisions_ind(dataframe):
    a = np.unique(dataframe['ID'])
    for t in a:
        b = dataframe[dataframe['ID'] == t]
        for k in b:
            for i in range(1, 5):

o = revisions_ind(ind_spf_trim)

###############################################
                ## EXPORT ##
###############################################

mean_spf_trim.to_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.cvs', sep=',', index=True)
ind_spf_trim.to_csv('Documents/thesis/_replication/cleaned_data/ind_spf_trim.cvs', sep=',', index=True)
vintage_trim.to_csv('Documents/thesis/_replication/cleaned_data/vintage_trim.cvs', sep=',', index=True)






