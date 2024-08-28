docs = '''
Robustness check estimations
'''

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np

# Working dir
os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')

# Import
vintage_trim = pd.read_csv('cleaned_data/vintage_trim.csv')
mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf_trim = pd.read_csv('cleaned_data/ind_spf_trim.csv')
oilp = pd.read_csv('cleaned_data/oil_prices.csv')

# Setting date variable
mean_spf_trim['DATE'] = pd.to_datetime(mean_spf_trim['DATE'])
ind_spf_trim['DATE'] = pd.to_datetime(ind_spf_trim['DATE'])

###############################################
                ## Controls ##
###############################################


def inf_control(df, v, end_date):
    '''
    Lagged inflation control
    '''
    v['lt_3'] = v['t3'].shift(1)
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    v = v.loc[(v['DATE'] >= '1981-12-31') & (v['DATE'] <= end_date)]
    L_pi = v['lt_3']
    revisions = df[f'r_t3']
    x = np.column_stack((L_pi, revisions))
    x = sm.add_constant(x)
    errors = df[f'e_t3']
    initial = sm.OLS(errors, x).fit()
    regs = initial.get_robustcov_results(cov_type='HAC', maxlags=None)
    return regs


def oil_control(df, oil, end_date):
    '''
    One quarter lagged log change in oil price control
    '''
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    oil = oil.loc[(oil['DATE'] >= '1981-12-31') & (oil['DATE'] <= end_date)]
    revisions = df[f'r_t3']
    L_oil = oil['pc_1q']
    x = np.column_stack((L_oil, revisions))
    x = sm.add_constant(x)
    errors = df[f'e_t3']
    initial = sm.OLS(errors, x).fit()
    regs = initial.get_robustcov_results(cov_type='HAC', maxlags=None)
    return regs


def E3R2_PLD(df, end_date):
    '''
    Regressing t3 errors on t2 revisions
    '''
    df = df.loc[(df['DATE'] >= '1981-12-31') & (df['DATE'] <= end_date)]
    revisions = df['r_t2']
    revisions = sm.add_constant(revisions)
    errors = df[f'e_t3']
    regs = sm.OLS(errors, revisions).fit(cov_type='cluster', cov_kwds = {"groups":df['ID']})
    return regs


def IV_OIL_INF(df, oil, v, end_date):
    '''
    2 quarters of lagged log change in oil prices IV estimation
    '''
    v['lt_3']= ((v['t']/400 + 1)*(v['t1']/400 + 1)*(v['t2']/400 + 1)*(v['t3']/400 + 1)-1)
    L_pi = v['lt_3'].shift(1)
    df = df.loc[(df['DATE'] >= '1986-09-30') & (df['DATE'] <= end_date)]
    oil = oil.loc[(oil['DATE'] >= '1986-09-30') & (oil['DATE'] <= end_date)]
    v = v.loc[(v['DATE'] >= '1986-09-30') & (v['DATE'] <= end_date)]
    L_pi = v['lt_3']
    revisions = df[f'r_t3']
    x = np.column_stack((revisions, L_pi))
    x = sm.add_constant(x)
    errors = df[f'e_t3']
    L1_oil = oil['pc_1q']
    L2_oil = oil['pc_2q']
    w = np.column_stack((L_pi, L1_oil, L2_oil))
    w = sm.add_constant(w)
    initial = IV2SLS(errors, x, instrument = w).fit()
    reg = initial.get_robustcov_results(cov_type='HAC', maxlags=None)
    return reg


date = '2020-06-30'  # Set desired end date for estimates
regs = [inf_control(mean_spf_trim, vintage_trim, date), oil_control(mean_spf_trim, oilp, date), E3R2_PLD(ind_spf_trim, date), IV_OIL_INF(mean_spf_trim, oilp, vintage_trim, date)]