import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

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

mean_spf_trim = mean_spf_trim.loc[(mean_spf_trim['DATE'] >= '1981-12-31') & (mean_spf_trim['DATE'] <= '2023-03-31')]  # Filter data

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
ind_spf_trim = ind_spf_trim.loc[(ind_spf_trim['DATE'] >= '1981-12-31') & (ind_spf_trim['DATE'] <= '2023-03-31')]  # Filter data

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

###############################################
                ## EXPORT ##
###############################################

mean_estimations.to_csv('output/mean_estimations.csv', sep=',', index=True)
ind_est_pld.to_csv('output/ind_est_pld.csv', sep=',', index=True)
ind_est_fe.to_csv('output/ind_est_fe.csv', sep=',', index=True)
ind_est_fe2.to_csv('output/ind_est_fe2.csv', sep=',', index=True)