import pandas as pd
import numpy as np
from _001cleaning import filter_date

###############################################
                ## Revisions ##
###############################################
spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.cvs')
spf = filter_date(spf, '1981_Q3', '2016_Q4')

means = []
for n in range(0, 2):
    means.append((spf[f'cpi_ft{n}'] - spf[f'cpi_ft{n+1}'].shift(1+n)).mean())
    
tot = np.mean(np.array(means))