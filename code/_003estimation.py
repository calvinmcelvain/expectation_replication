import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

vintage_trim = pd.read_csv('cleaned_data/vintage_trim.csv')

mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf_trim = pd.read_csv('cleaned_data/ind_spf_trim.csv')

mean_spf_trim['DATE'] = pd.to_datetime(mean_spf_trim['Unnamed: 0'])
mean_spf_trim = mean_spf_trim.drop(['Unnamed: 0'], axis=1)

ind_spf_trim['DATE'] = pd.to_datetime(ind_spf_trim['Unnamed: 0'])
ind_spf_trim = ind_spf_trim.drop(['Unnamed: 0'], axis=1)

###############################################
                ## ESTIMATION ##
###############################################

#####################
## Mean Forecasts ##
#####################

mean_spf_trim = mean_spf_trim.loc[(mean_spf_trim['DATE'] >= '1981-12-31') & (mean_spf_trim['DATE'] <= '2016-12-31')]  # Filter data

def est_table(df):
    est_table = pd.DataFrame(index=['const', 'beta_1'])
    for j in range(4):
        revisions = df[f'r_t{j}']
        revisions = sm.add_constant(revisions)
        errors = df[f'e_t{j}']
        initial = sm.OLS(errors, revisions).fit()
        reg = initial.get_robustcov_results(cov_type='HAC', maxlags=1, hasconst=True)
        est_table[f'coef_t{j}'] = reg.params
        est_table[f'std_err{j}'] = reg.bse
        est_table[f'tval{j}'] = reg.tvalues
        est_table[f'pval{j}'] = reg.pvalues
        est_table[f'nobs{j}'] = initial.nobs
    return est_table

mean_estimations = est_table(mean_spf_trim)

#####################
 ## Ind Forecasts ##
#####################

### Pooled OLS ###
ind_spf_trim = ind_spf_trim.loc[(ind_spf_trim['DATE'] >= '1981-12-31') & (ind_spf_trim['DATE'] <= '2016-12-31')]  # Filter data

ind_spf_trim = ind_spf_trim.dropna(subset='r_t1')
ind_est_pld = est_table(ind_spf_trim)

### ID Fixed Effects ###
def est_table_fe(df):
    est_table = pd.DataFrame(index=['const', 'beta_1'])
    for j in range(4):
        x = np.column_stack((np.ones(len(df)), df[f'r_t{j}'], pd.get_dummies(df['ID'])))
        y = np.asarray(df[f'e_t{j}'])
        initial = sm.OLS(y, x).fit()
        reg = initial.get_robustcov_results(cov_type='HAC', maxlags=1, hasconst=True)
        est_table[f'coef_t{j}'] = reg.params[:2]
        est_table[f'std_err{j}'] = reg.bse[:2]
        est_table[f'tval{j}'] = reg.tvalues[:2]
        est_table[f'pval{j}'] = reg.pvalues[:2]
        est_table[f'nobs{j}'] = initial.nobs
    return est_table

ind_est_fe = est_table_fe(ind_spf_trim)

### Two-way Fixed Effects ###
def est_table_fe2(df):
    est_table = pd.DataFrame(index=['const', 'beta_1'])
    for j in range(4):
        x = np.column_stack((np.ones(len(df)), df[f'r_t{j}'], pd.get_dummies(df['ID']), pd.get_dummies(ind_spf_trim['DATE'])))
        y = np.asarray(df[f'e_t{j}'])
        initial = sm.OLS(y, x).fit()
        reg = initial.get_robustcov_results(cov_type='HAC', maxlags=1, hasconst=True)
        est_table[f'coef_t{j}'] = reg.params[:2]
        est_table[f'std_err{j}'] = reg.bse[:2]
        est_table[f'tval{j}'] = reg.tvalues[:2]
        est_table[f'pval{j}'] = reg.pvalues[:2]
        est_table[f'nobs{j}'] = initial.nobs
    return est_table

ind_est_fe2 = est_table_fe2(ind_spf_trim)

#####################
    ## AR(1) ##
#####################

vintage_trim = vintage_trim.loc[(vintage_trim['DATE'] >= '1965-06-30') & (vintage_trim['DATE'] <= '2016-12-31')]  # Filter data

def ar_j(df):
    ar_table = pd.DataFrame(index=['phi'])
    grwth_t = df['t0'][2:]
    grwth_t0 = df['t0'].shift(1)[2:]
    reg = sm.OLS(grwth_t, grwth_t0).fit()
    ar_table['coef'] = reg.params.iloc[0]
    ar_table['std_err'] = reg.bse.iloc[0]
    return ar_table

ar_table = ar_j(vintage_trim)

###############################################
            ## MODEL PARAMETERS ##
###############################################

def params(dfm, dfp, dff, dff2, ar):
    params = pd.DataFrame(index=['ols/pld', 'fe', 'fe2'])
    params.loc['ols/pld', 'lambda'] = dfm.loc['beta_1', 'coef_t3'] / (1 + dfm.loc['beta_1', 'coef_t3'])
    params.loc['ols/pld', 'G'] = 1 / (1 + dfm.loc['beta_1', 'coef_t3'])
    pldc = dfp.iloc[1]['coef_t3']
    fec = dff.iloc[1]['coef_t3']
    fe2c = dff2.iloc[1]['coef_t3']
    ar1 = ar.iloc[0]['coef']
    params.loc['ols/pld', 'Theta'] = (-((2 * pldc) + 1) + np.sqrt(((2 * pldc) + 1)**2 - 4 * (pldc + (pldc * ar1**2) + 1) * pldc)) / (2 * (pldc + (pldc * ar1**2) + 1))
    params.loc['fe', 'Theta'] = (-((2 * fec) + 1) + np.sqrt(((2 * fec) + 1)**2 - 4 * (fec + (fec * ar1**2) + 1) * fec)) / (2 * (fec + (fec * ar1**2) + 1))
    params.loc['fe2', 'Theta'] = (-((2 * fe2c) + 1) + np.sqrt(((2 * fe2c) + 1)**2 - 4 * (fe2c + (fe2c * ar1**2) + 1) * fe2c)) / (2 * (fe2c + (fe2c * ar1**2) + 1))
    return params

parameters = params(mean_estimations, ind_est_pld, ind_est_fe, ind_est_fe2, ar_table)

###############################################
                ## EXPORT ##
###############################################

mean_estimations.to_csv('output/mean_estimations.csv', sep=',', index=True)
ind_est_pld.to_csv('output/ind_est_pld.csv', sep=',', index=True)
ind_est_fe.to_csv('output/ind_est_fe.csv', sep=',', index=True)
ind_est_fe2.to_csv('output/ind_est_fe2.csv', sep=',', index=True)
ar_table.to_csv('output/ar_estimations.csv', sep=',', index=True)