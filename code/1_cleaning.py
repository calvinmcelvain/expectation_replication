import pandas as pd

def filter_date(data, a, b):
    c = data[(data['yrq'] >= a)]
    d = c[(c['yrq'] <= b)]
    return d

###############################################
## SPF DATA ##
###############################################
mean_spf = pd.read_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv")
mean_spf.to_csv("Documents/thesis/_replication/data/spf_mean_cpi.csv", sep=',', index=False)  # Original file had CRLF endings, changed to LF for Git

mean_spf_trim = mean_spf
mean_spf_trim['yrq'] = mean_spf_trim['YEAR'].astype(str) + '_Q' + mean_spf_trim['QUARTER'].astype(str)
mean_spf_trim.set_index('yrq', inplace=True)
mean_spf_trim = mean_spf_trim.fillna("")
mean_spf_trim = mean_spf_trim.drop(['YEAR', 'QUARTER', 'CPI6','CPIA', 'CPIB', 'CPIC'], axis=1)
mean_spf_trim = mean_spf_trim.rename(columns={'CPI1':'cpi_t+1', 'CPI2': 'cpi_t+2', 'CPI3': 'cpi_t+3', 'CPI4': 'cpi_t+4', 'CPI5': 'cpi_t+5'})


###############################################
## Vintage CPI DATA ##
###############################################
vintage = pd.read_csv("Documents/thesis/_replication/data/vintage_cpi.csv")
vintage.to_csv("Documents/thesis/_replication/data/vintage_cpi.csv", sep=',', index=False)    # Original file had CRLF endings, changed to LF for Git

vintage_trim = vintage
vintage_trim['DATE'] = [date.replace(':', '_') for date in vintage_trim['DATE']]
vintage_trim = vintage_trim.rename(columns={'DATE': 'yrq'})
vintage_trim = vintage_trim.fillna("")
vintage_trim.set_index('yrq', inplace=True)

def rename_col_vintage(column_name):
    if column_name.startswith('RCON'):
        parts = column_name.split('Q')
        year = int(parts[0][4:6])
        if year >= 50:
            year += 1900
        else:
            year += 2000
        return f"base_{year}_Q{parts[1]}"
    else:
        return column_name

vintage_trim = vintage_trim.rename(columns=rename_col_vintage)


###############################################
## EXPORT ##
###############################################
mean_spf_trim.to_csv('Documents/thesis/_replication/cleaned_data/spf_mean_trim.cvs', sep=',', index=True)
vintage_trim.to_csv('Documents/thesis/_replication/cleaned_data/vintage_trim.cvs', sep=',', index=True)