import os
import pandas as pd

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

mean_spf = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf = pd.read_csv('cleaned_data/ind_spf_trim.csv')
vintage = pd.read_csv('cleaned_data/vintage_trim.csv')

mean_spf = mean_spf.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])
vintage = vintage.set_index('DATE')

###############################################
                  ## AR(1) ##
###############################################
