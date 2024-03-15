import pandas as pd

###############################################
            ## GENERAL CLEANING ##
###############################################

## SPF DATA ##
mean_spf = pd.read_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv")
mean_spf.to_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git

mean_spf_trim = mean_spf
mean_spf_trim['yrq'] = mean_spf_trim['YEAR'].astype(str) + '_Q' + mean_spf_trim['QUARTER'].astype(str)
mean_spf_trim = mean_spf_trim.fillna("")
mean_spf_trim = mean_spf_trim.drop(['YEAR', 'QUARTER', 'CPI6','CPIA', 'CPIB', 'CPIC'], axis=1)
mean_spf_trim = mean_spf_trim.rename(columns={'CPI1':'cpi_ft+1', 'CPI2': 'cpi_ft+2', 'CPI3': 'cpi_ft+3', 'CPI4': 'cpi_ft+4', 'CPI5': 'cpi_ft+5'})

def filter_date(data, a, b):
    c = data[(data['yrq'] >= a)]
    if b != 0:
        d = c[(c['yrq'] <= b)]
        return d
    else:
        return c

mean_spf_trim = filter_date(mean_spf_trim, '1981_Q3', 0)

## Vintage CPI DATA ##
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
vintage_trim.set_index('yrq', inplace=True) # The indexing trifecta

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

for var in vintage_trim.columns:
    vintage_trim[f'{var}'].astype(float)

###############################################
            ## CALCULATING GROWTH RATES ##
###############################################



###############################################
                ## EXPORT ##
###############################################

mean_spf_trim.to_csv('Documents/thesis/_replication/cleaned_data/spf_mean_trim.cvs', sep=',', index=True)
vintage_trim.to_csv('Documents/thesis/_replication/cleaned_data/vintage_trim.cvs', sep=',', index=True)





