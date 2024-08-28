docs = '''
- Estimates for OLS, Pooled OLS, Fixed-effects, and IV specifications.
- AR(1) estimates
'''

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np

# Working dir
os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')

# Importing 
oilp = pd.read_csv('cleaned_data/vintage_trim.csv')
vintage_trim = pd.read_csv('cleaned_data/vintage_trim.csv')
mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf_trim = pd.read_csv('cleaned_data/ind_spf_trim.csv')

# Setting date
mean_spf_trim['DATE'] = pd.to_datetime(mean_spf_trim['DATE'])
ind_spf_trim['DATE'] = pd.to_datetime(ind_spf_trim['DATE'])


###############################################
                ## ESTIMATION ##
###############################################


### OLS ###
def OLS(df, end_date):
    '''
    Function to return mean OLS estimates for time horizons t0 - t3
    '''
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    regs = []
    for j in range(4):
        revisions = df[f'r_t{j}']
        revisions = sm.add_constant(revisions)
        errors = df[f'e_t{j}']
        initial = sm.OLS(errors, revisions).fit()
        regs.append(initial.get_robustcov_results(cov_type='HAC', maxlags=None))
    return regs


### PLD OLS ###
def OLS_PLD(df, end_date):
    '''
    Function to return individual-level pooled OLS estimates for time horizons t0 - t3
    '''
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    regs = []
    for j in range(4):
        revisions = df[f'r_t{j}']
        revisions = sm.add_constant(revisions)
        errors = df[f'e_t{j}']
        regs.append(sm.OLS(errors, revisions).fit(cov_type='cluster', cov_kwds = {"groups":df['ID']}))
    return regs


### ID FE ###
def ID_FE(df, end_date):
    '''
    Function to return mean fixed-effect(time) estimates for time horizons t0 - t3
    '''
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    regs = []
    for j in range(4):
        x = np.column_stack((np.ones(len(df)), df[f'r_t{j}'], pd.get_dummies(df['ID'], drop_first=True, dtype=float)))
        y = np.asarray(df[f'e_t{j}'])
        regs.append(sm.OLS(y, x).fit(cov_type='cluster', cov_kwds = {"groups":df['ID']}))
    return regs


### AR Estimates ###
def AR(df, end_date):
    '''
    AR(1) estimates for t3
    '''
    df = df.loc[(df['DATE'] >= '1965-06-30') & (df['DATE'] <= end_date)]  # Filter data
    grwth_t = df['t3']
    reg = AutoReg(grwth_t, 1).fit()
    return reg


### Parameter Estimates ###
def params(ols, pldols, fe, ar1):
    '''
    Calculates model parameter estimates using above estimates
    '''
    params = []
    params.append(ols / (1 + ols))
    params.append(1 / (1 + ols))
    params.append((-((2 * pldols) + 1) + np.sqrt(((2 * pldols) + 1)**2 - 4 * (pldols + (pldols * ar1**2) + 1) * pldols)) / (2 * (pldols + (pldols * ar1**2) + 1)))
    params.append((-((2 * fe) + 1) + np.sqrt(((2 * fe) + 1)**2 - 4 * (fe + (fe * ar1**2) + 1) * fe)) / (2 * (fe + (fe * ar1**2) + 1)))
    return params


def compute_regs(date, mean, ind):
    '''
    Computes all regression estimates and calculates model parameter estimates for each horizon.
    Returns a list of regression estimates and model parameter estimates.
    '''
    mean_regs = OLS(mean, f'{date}')
    ind_regs = OLS_PLD(ind, f'{date}')
    ind_regs_fe = ID_FE(ind, f'{date}')
    ar_1 = AR(vintage_trim, f'{date}')
    regs = [mean_regs, ind_regs, ind_regs_fe, ar_1]
    parameters = params(mean_regs[3].params[1], ind_regs[3].params[1], ind_regs_fe[3].params[1], ar_1.roots)
    return regs, parameters


date = '2022-12-31' # Set desired end date for estimates
regs, parameters = compute_regs(date, mean_spf_trim, ind_spf_trim)

## OIL INSTRUMENT ##
def IV_OIL_OLS(df, oil, end_date):
    '''
    IV estimation using 2 lags of log oil price changes as instruments
    '''
    df = df.loc[(df['DATE'] >= '1986-09-30') & (df['DATE'] <= end_date)]
    oil = oil.loc[(oil['DATE'] >= '1986-09-30') & (oil['DATE'] <= end_date)]
    revisions = df[f'r_t3']
    x = sm.add_constant(revisions)
    errors = df[f'e_t3']
    L1_oil = oil['pc_1q']
    L2_oil = oil['pc_2q']
    w = np.column_stack((L1_oil, L2_oil))
    w = sm.add_constant(w)
    initial = IV2SLS(errors, x, instrument = w).fit()
    reg = initial.get_robustcov_results(cov_type='HAC', maxlags=None)
    return reg


iv_reg = IV_OIL_OLS(mean_spf_trim, oilp, date)