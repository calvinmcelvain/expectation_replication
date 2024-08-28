doc = '''
- General SPF CPI data & vintage data cleaning
- Creating seperate datasets for mean and individual SPF CPI forecasts
- CPI forecast growth, revision, and error calculations
- Oil price change calculations
'''

import os
import pandas as pd
import numpy as np

# Working dir.
os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')

###############################################
            ## GENERAL CLEANING ##
###############################################


#####################
    ## SPF DATA ##
#####################


### Individual Forecasts ###

# Importing raw SPF CPI forecast data (individual)
ind_spf_trim = pd.read_csv("data/spf_ind_cpi.csv")
ind_spf_trim.to_csv("data/spf_ind_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git

# Creating date variable and setting df to be panel data
ind_spf_trim['DATE'] = pd.to_datetime(ind_spf_trim['YEAR'].astype(str) + '-' + (ind_spf_trim['QUARTER'] * 3).astype(str), format='%Y-%m') + pd.offsets.MonthEnd(0)
ind_spf_trim = ind_spf_trim.set_index(['DATE', 'ID'])  # Panel data

# Dropping data pre-1981Q3, unused vars, and forecasts with missing forecast horizons
ind_spf_trim = ind_spf_trim['1981-09-01':]
ind_spf_trim = ind_spf_trim.drop(['YEAR', 'QUARTER', 'CPIA', 'CPIB', 'CPIC', 'INDUSTRY'], axis=1)
ind_spf_trim = ind_spf_trim.dropna(subset=['CPI1', 'CPI2', 'CPI3', 'CPI4', 'CPI5', 'CPI6'])

# Dropping forecasters with <10 forecasts
forecast_counts = ind_spf_trim.groupby('ID').size()
ind_spf_trim = ind_spf_trim[ind_spf_trim.index.get_level_values('ID').isin(forecast_counts[forecast_counts >= 10].index)]

# Winsorizing forecasts
def winsorizing_forecasts(df):
    '''
    Winsoring each forecast horizon to 5 IQR
    '''
    for date in df.index.get_level_values('DATE').unique():
        for i in range(2, 7):
            column_name = f'CPI{i}'
            q25 = np.quantile(df.loc[date, column_name], 0.25)
            q50 = np.quantile(df.loc[date, column_name], 0.50)
            q75 = np.quantile(df.loc[date, column_name], 0.75)
            uf = q50 + 5 * (q75 - q25)
            lf = q50 - 5 * (q75 - q25)
            df.loc[date, column_name] = df.loc[(date, slice(None)), column_name].where((df.loc[(date, slice(None)), column_name] >= lf) & (df.loc[(date, slice(None)), column_name] <= uf), np.nan)
    df = df.dropna(subset=['CPI1', 'CPI2', 'CPI3', 'CPI4', 'CPI5', 'CPI6'])
    return df


ind_spf_trim = winsorizing_forecasts(ind_spf_trim)


### Mean Forecasts ###

# Mean aggregating individual forecasts
mean_spf_trim = ind_spf_trim.groupby(level='DATE').mean()


#####################
 ## VINTAGE DATA ##
#####################


# Importing vintage data
vintage_trim = pd.read_csv("data/vintage_cpi.csv")
vintage_trim.to_csv("data/vintage_cpi.csv", sep=',', index=False)    # Original file had CRLF endings, changed to LF for Git

# Creating date variable
vintage_trim['DATE'] = pd.to_datetime(vintage_trim['DATE'], format='%Y:%m')
vintage_trim.set_index('DATE', inplace=True)

# Mean aggregating to quarterly to match forecasts (data is collected monthly)
vintage_trim = vintage_trim.resample('Q').mean()

# Dropping oberservations pre-1965Q1
vintage_trim = vintage_trim['1965-03-01':]

# Dropping empty vintages
def keep_cols(dfcols, a, b, di):
    '''
    Vintage data has columns with empty vintages pre-1994Q3. This is a function to drop those columns.
    '''
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


keep_cols = keep_cols(vintage_trim.columns, 64, 95, ['CPI94Q3', 'CPI94Q4'])
vintage_trim = vintage_trim.drop(columns=keep_cols)

# Adding recession variable to all datasets
def recessions(df):
    '''
    Recession time periods come directly from NBER
    '''
    df['rec'] = 0
    df.loc['1980-06-30':'1981-06-30', 'rec'] = 1
    df.loc['1989-12-31':'1991-03-31', 'rec'] = 1
    df.loc['2001-03-31':'2001-09-30', 'rec'] = 1
    df.loc['2007-12-31':'2009-06-30', 'rec'] = 1
    df.loc['2020-03-31':'2020-06-30', 'rec'] = 1
    return df


vintage_trim = recessions(vintage_trim)
ind_spf_trim = recessions(ind_spf_trim)
mean_spf_trim = recessions(mean_spf_trim)

#####################
 ## OIL DATA ##
#####################


# Importing 
oilp = pd.read_csv('data/wti_spot_oil.csv')

# Renaming price column
oilp = oilp.rename(columns={'WTISPLC': 'price'})

# Offsetting date to match SPF date setting above
oilp['DATE'] = pd.to_datetime(oilp['DATE']) - pd.offsets.MonthEnd(1)


###############################################
        ## PRELIMINARY CALCULATIONS ##
###############################################


#####################
 ## ACTUAL GROWTH ##
#####################


### Actuals ###
def actual_growth(df):
    '''
    Adding the realized CPI in the forecasted time period.
    Vintages start at 1994Q3, thus for realized CPI for horizons from 1981Q3-1994Q3 realized CPI comes from 1994Q3 vintage
    '''
    for i in vintage_trim.index[1:]:
        column_name = ('CPI' + str(i.year)[-2:] + 'Q' + str((i.month - 1) // 3 + 1)) if i > pd.Timestamp('1994-09-30') else 'CPI94Q3'
        df.loc[i, 't'] = vintage_trim.loc[i - pd.offsets.QuarterEnd(), column_name]
    
    for i in df.index[1:]:
        for j in range(0, 5):
            next_quarters = i + pd.offsets.QuarterEnd(j+1)
            if next_quarters in df.index:   # matching vintages to information sets
                df.loc[i, f't{j}'] = ((df.loc[next_quarters, 't'] / df.loc[i, 't']) - 1)


actual_growth(vintage_trim)


### Forecasts ###
def cpi_growth(df):
    '''
    Calculating forecasted CPI growth for t0 to t4 (t3 is forecasted annual CPI growth)
    '''
    for i in df.index:
        df.loc[i, 'f_t0'] = df.loc[i, 'CPI2']/400
        for j in range(1,5):
            df.loc[i, f'f_t{j}'] = (df.loc[i, f'f_t{j-1}']+1) * (df.loc[i, f'CPI{j+2}']/400 +1) - 1


cpi_growth(mean_spf_trim)
cpi_growth(ind_spf_trim)


#####################
   ## REVISIONS ##
#####################


### Mean Forecasts ###
def mean_revisions(df):
    '''
    Calculating mean forecast revisions
    '''
    for i in df.index:
        for j in range(0, 4):
            if (i - pd.offsets.QuarterEnd()) in df.index:
                df.loc[i, f'r_t{j}'] = df.loc[i, f'f_t{j}'] - ((df.loc[i - pd.offsets.QuarterEnd(), f'f_t{j + 1}'] + 1) / (df.loc[i - pd.offsets.QuarterEnd(), 'f_t0'] + 1) - 1)   # Only revisions since t-1


mean_revisions(mean_spf_trim)


### Individual Forecasts ###
def ind_revisions(df):
    '''
    Calculating individual forecast revisions
    '''
    for d, i in df.index:
        for j in range(0, 4):
            if (d- pd.offsets.QuarterEnd(), i) in df.index:
                previous_quarter = d - pd.offsets.QuarterEnd()
                df.loc[(d, i), f'r_t{j}'] = df.loc[(d, i), f'f_t{j}'] - ((df.loc[(previous_quarter, i), f'f_t{j + 1}'] + 1) / (df.loc[(previous_quarter, i), 'f_t0'] + 1) - 1)    # Only revisions since t-1


ind_revisions(ind_spf_trim)


#####################
    ## ERRORS ##
#####################


### Mean Forecasts ###
def mean_errors(df, v):
    '''
    Calculating mean CPI forecasting errors using realized CPI
    '''
    for i in df.index:
        for j in range(0, 5):
            df.loc[i, f'e_t{j}'] = v.loc[i, f't{j}'] - df.loc[i, f'f_t{j}']
        

mean_errors(mean_spf_trim, vintage_trim)


### Individual Forecasts ###
def ind_errors(df, v):
    '''
    Calculating individual-level CPI forecasting errors using realized CPI
    '''
    for d, i in df.index:
        for j in range(0, 5):
            df.loc[(d, i), f'e_t{j}'] = v.loc[d, f't{j}'] - df.loc[(d, i), f'f_t{j}']


ind_errors(ind_spf_trim, vintage_trim)


############################
   ## OIL PRICE CHANGE ##
############################


# Log change in oil prices for 2 lagged quarters
oilp['pc_1q'] = np.log(oilp['price'].shift(1) / oilp['price'].shift(2))
oilp['pc_2q'] = np.log(oilp['price'].shift(2) / oilp['price'].shift(3))


###############################################
                ## EXPORT ##
###############################################


mean_spf_trim.to_csv('cleaned_data/mean_spf_trim.csv', sep=',', index=True)
ind_spf_trim.to_csv('cleaned_data/ind_spf_trim.csv', sep=',', index=True)
vintage_trim.to_csv('cleaned_data/vintage_trim.csv', sep=',', index=True)
oilp.to_csv('cleaned_data/oil_prices.csv')