import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')    # Working dir

mean_spf = pd.read_csv('cleaned_data/mean_spf_trim.csv')
ind_spf = pd.read_csv('cleaned_data/ind_spf_trim.csv')
vintage = pd.read_csv('cleaned_data/vintage_trim.csv')

mean_spf = mean_spf.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])
vintage = vintage.set_index('DATE')

###############################################
            ## SCATTER PLOTS ##
###############################################

### Revisions - Errors ###
fig, (axm, axi) = plt.subplots(nrows=1, sharey=True, ncols=2, figsize=(8, 4))

# Mean
x = mean_spf['r_t3']
y = mean_spf['e_t3']
axm.scatter(x, y, c='navy', alpha=1)
axm.set_xlabel('Mean Revisions', fontsize=12)
axm.set_ylabel('Errors', fontsize=12)

# Individual
x2 = ind_spf['r_t3']
y2 = ind_spf['e_t3']
axi.scatter(x2, y2, c='black', alpha=1)
axi.set_xlabel('Individual Revisions', fontsize=12)

plt.show()

fig.savefig('output/re_scat.png')

###  ###