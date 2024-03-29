import os
import pandas as pd

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

###############################################
            ## GENERAL CLEANING ##
###############################################

#####################
    ## SPF DATA ##
#####################

### Mean Forecasts ###
mean_spf = pd.read_csv("data/spf_mean_cpi.csv")
mean_spf.to_csv("data/spf_mean_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git
mean_spf_trim = mean_spf

mean_spf_trim = mean_spf_trim.set_index(pd.to_datetime(mean_spf_trim['YEAR'].astype(str) + '-' + (mean_spf_trim['QUARTER'] * 3).astype(str), format='%Y-%m') + pd.offsets.MonthEnd(0))

mean_spf_trim = mean_spf_trim.drop(['YEAR', 'QUARTER', 'CPIA', 'CPIB', 'CPIC'], axis=1)
mean_spf_trim = mean_spf_trim['1981-09-01':]

### Individual Forecasts ###
ind_spf = pd.read_csv("data/spf_ind_cpi.csv")
ind_spf.to_csv("data/spf_ind_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git
ind_spf_trim = ind_spf

ind_spf_trim = ind_spf_trim.set_index([pd.to_datetime(ind_spf_trim['YEAR'].astype(str) + '-' + (ind_spf_trim['QUARTER'] * 3).astype(str), format='%Y-%m') + pd.offsets.MonthEnd(0), 'ID'])  # Panel data
ind_spf_trim = ind_spf_trim.dropna(axis=0, thresh=8)
ind_spf_trim = ind_spf_trim.drop(['YEAR', 'QUARTER', 'CPIA', 'CPIB', 'CPIC'], axis=1)
ind_spf_trim = ind_spf_trim['1981-09-01':]

#####################
 ## VINTAGE DATA ##
#####################

vintage = pd.read_csv("data/vintage_cpi.csv")
vintage.to_csv("data/vintage_cpi.csv", sep=',', index=False)    # Original file had CRLF endings, changed to LF for Git
vintage_trim = vintage

vintage_trim['DATE'] = pd.to_datetime(vintage_trim['DATE'], format='%Y:%m')
vintage_trim.set_index('DATE', inplace=True)
vintage_trim = vintage_trim.resample('Q').mean()    # Data is collected monthly, mean aggregating to quarterly to match forecasts
vintage_trim = vintage_trim['1965-03-01':]

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
        ## PRELIMINARY CALCULATIONS ##
###############################################

#####################
 ## ACTUAL GROWTH ##
#####################

### Actuals ###
def actual_growth(df):
    for i in vintage_trim.index[1:]:
        column_name = ('CPI' + str(i.year)[-2:] + 'Q' + str((i.month - 1) // 3 + 1)) if i > pd.Timestamp('1994-09-30') else 'CPI94Q3'
        df.loc[i, 't'] = vintage_trim.loc[i - pd.offsets.QuarterEnd(), column_name]
    
    for i in df.index[1:]:
        for j in range(1, 6):
            next_quarters = i + pd.offsets.QuarterEnd(j)
            if next_quarters in df.index:
                df.loc[i, f't{j}'] = ((df.loc[next_quarters, 't'] / df.loc[i, 't']) - 1)
actual_growth(vintage_trim)

### Forecasts ###
def cpi_growth(df):
    for i in df.index:
        df.loc[i, 'f_t0'] = df.loc[i, 'CPI2']/400
        for j in range(1,5):
            df.loc[i, f'f_t{j}'] = (df.loc[i, f'f_t{j-1}']+1) * (df.loc[i, f'CPI{j+2}']/400 +1) -1
            
cpi_growth(mean_spf_trim)
cpi_growth(ind_spf_trim)

#####################
   ## Revisions ##
#####################

### Mean Forecasts ###
def revisions(df):
    for i in df.index:
        for j in range(0, 4):
            if (i - pd.offsets.QuarterEnd()) in df.index:
                df.loc[i, f'r_t{j}'] = df.loc[i, f'f_t{j}'] - ((df.loc[i - pd.offsets.QuarterEnd(), f'f_t{j + 1}'] + 1) / (df.loc[i - pd.offsets.QuarterEnd(), 'f_t0'] + 1) - 1)    # Only revisions since t-1

revisions(mean_spf_trim)

### Individual Forecasts ###
def revisions_ind(df):
    for d, i in df.index:
        for j in range(0, 4):
            if (d- pd.offsets.QuarterEnd(), i) in df.index:
                previous_quarter = d - pd.offsets.QuarterEnd()
                df.loc[(d, i), f'r_t{j}'] = df.loc[(d, i), f'f_t{j}'] - ((df.loc[(previous_quarter, i), f'f_t{j + 1}'] + 1) / (df.loc[(previous_quarter, i), 'f_t0'] + 1) - 1)    # Only revisions since t-1

revisions_ind(ind_spf_trim)

#####################
    ## Errors ##
#####################

### Mean Forecasts ###
def errors(df, v):
    for i in df.index:
        for j in range(0, 5):
            df.loc[i, f'e_t{j}'] = v.loc[i, f't{j+1}'] - df.loc[i, f'f_t{j}']

errors(mean_spf_trim, vintage_trim)

### Individual Forecasts ###
def errors_ind(df, v):
    for d, i in df.index:
        for j in range(0, 5):
            df.loc[(d, i), f'e_t{j}'] = v.loc[d, f't{j+1}'] - df.loc[(d, i), f'f_t{j}']

errors_ind(ind_spf_trim, vintage_trim)

###############################################
                ## EXPORT ##
###############################################

mean_spf_trim.to_csv('cleaned_data/mean_spf_trim.csv', sep=',', index=True)
ind_spf_trim.to_csv('cleaned_data/ind_spf_trim.csv', sep=',', index=True)
vintage_trim.to_csv('cleaned_data/vintage_trim.csv', sep=',', index=True)






