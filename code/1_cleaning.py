import numpy as np
import pandas as pd
import statsmodels.api as sm

mean_spf = pd.read_csv("data/spf_mean_cpi.csv")

for y,q in mean_spf['YEAR'], mean_spf['QUARTER']:
    mean_spf['yrq'] = f"{y}_"

mean_spf = mean_spf[(mean_spf['YEAR'] >= 1981)]

