import os
import pandas as pd
import statsmodels.api as sm

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf = pd.read_csv('cleaned_data/ind_spf_trim.csv')

mean_spf_trim = mean_spf_trim.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])

###############################################
                ## ESTIMATION ##
###############################################

mean_spf_trim = mean_spf_trim['1981-12-31':'2023-03-31']  # Filter data

### Mean Forecasts ###
revisions3 = mean_spf_trim['r_t3']
revisions3 = sm.add_constant(revisions3)
errors3 = mean_spf_trim['e_t3']

initial = sm.OLS(errors3, revisions3).fit()
reg = initial.get_robustcov_results(cov_type='HAC',maxlags=1)

print(reg.summary())

### Individual Forecasts ###
