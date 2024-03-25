import pandas as pd
import statsmodels.api as sm
import numpy as np

mean_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.csv')
ind_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/ind_spf_trim.csv')

mean_spf = mean_spf.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])

###############################################
                ## ESTIMATION ##
###############################################

### Mean Forecasts ###
revisions3 = mean_spf[1:]['r_t3']
errors3 = mean_spf[1:]['e_t3']

initial = sm.OLS(errors3, revisions3).fit()
reg = initial.get_robustcov_results(cov_type='HAC',maxlags=1)

print(reg.summary())