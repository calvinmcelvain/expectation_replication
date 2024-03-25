import pandas as pd

mean_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/mean_spf_trim.cvs')
ind_spf = pd.read_csv('Documents/thesis/_replication/cleaned_data/ind_spf_trim.cvs')

mean_spf = mean_spf.set_index('Unnamed: 0')
ind_spf = ind_spf.set_index(['Unnamed: 0', 'ID'])


###############################################
                ## ERRORS ##
###############################################



###############################################
            ## REVISIONS ##
###############################################