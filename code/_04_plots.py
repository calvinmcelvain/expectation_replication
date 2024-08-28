docs = '''
Plots and such.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import _03_estimation as estimations

# Working dir
os.chdir('/Users/fogellmcmuffin/Documents/thesis/_replication/')

#########################
      ## Plots ##
#########################


def rev_errs(regs, date, name=None, save=False, path='output/'):
   '''
   Plots both individual and mean forecasts w/ regression line.
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
   ind_spf_trim = pd.read_csv('cleaned_data/ind_spf_trim.csv')
   mean_spf_trim = mean_spf_trim.loc[(mean_spf_trim['DATE'] >= '1981-09-30') & (mean_spf_trim['DATE'] <= f'{date}')]
   ind_spf_trim = ind_spf_trim.loc[(ind_spf_trim['DATE'] >= '1981-09-30') & (ind_spf_trim['DATE'] <= f'{date}')]

   fig, (axm, axi) = plt.subplots(nrows=1, sharey=True, ncols=2, figsize=(8, 4))
   plt.ylim(-0.08, 0.08)

   xm = mean_spf_trim['r_t3'][1:]
   ym = mean_spf_trim['e_t3'][1:]

   axm.scatter(xm, ym, c='navy', alpha=1, s=20)
   model_m = regs[0][3].params[0] + regs[0][3].params[1] * np.linspace(-0.02, 0.02)
   axm.plot(np.linspace(-0.02, 0.02),  model_m, color='red', lw=1.5)

   axm.set_xlabel(r'$f_{t}\pi_{t+3} - f_{t-1}\pi_{t+3}$', fontsize=12)
   axm.set_ylabel(r'$f_{t}\pi_{t+3}-\pi_{t+3}$', fontsize=12)

   ### Individual ###
   xi = ind_spf_trim['r_t3']
   yi = ind_spf_trim['e_t3']
   model_p = regs[1][3].params[0] + regs[1][3].params[1] * np.linspace(-0.1, 0.1)
   model_fe = regs[2][3].params[0] + regs[2][3].params[1] * np.linspace(-0.1, 0.1)

   axi.scatter(xi, yi, c='blueviolet', alpha=1, s=20)
   axi.plot(np.linspace(-0.1, 0.1), model_p, label='OLS', color='red', lw=1.5)
   axi.plot(np.linspace(-0.1, 0.1), model_fe, label='ID FE', color='lime', lw=1.5)

   axi.set_xlabel(r'$f^{\ i}_{\ t}\pi_{t+3} - f^{\ i}_{\ t-1}\pi_{t+3}$', fontsize=12)
   axi.legend()

   plt.show()
   if save == True:
      fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)


def auto_reg(regs, date, name=None, save=False, path='output/'):
   '''
   Plots CPI vintage data for 4 horizons and plots AR(1) regression line
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   vintage_trim = pd.read_csv('cleaned_data/vintage_trim.csv')
   vintage_trim = vintage_trim.set_index(pd.to_datetime(vintage_trim['DATE']))

   vintage_trim = vintage_trim.loc[(vintage_trim['DATE'] >= '1965-06-30') & (vintage_trim['DATE'] <= date)]
   vintage_trim = vintage_trim[['t0', 't1', 't2', 't3', 't4', 'rec']]

   model0 = regs[3].params[0] + regs[3].params[1] * np.linspace(-0.025, 0.150)

   fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharey=True,)

   ax0.scatter(vintage_trim['t0']['1965-09-30':], vintage_trim['t0'].shift(1)['1965-09-30':], c='blue', label=r't = 1', alpha=1, s=20)
   ar = ax0.plot(np.linspace(-0.025, 0.150), model0, color="red", lw=1.5, label='AR(1)')
   ax0.set_ylabel(r'$\pi_{t-1}$', fontsize=10)
   ax0.legend(['t = 1'], fontsize=12)

   ax1.scatter(vintage_trim['t1']['1965-09-30':], vintage_trim['t1'].shift(1)['1965-09-30':], c='green', label='t = 2', alpha=1, s=20)
   ax1.plot(np.linspace(-0.025, 0.150), model0, color="red", lw=1.5)
   ax1.legend(['t = 2'], fontsize=12)

   ax2.scatter(vintage_trim['t2']['1965-09-30':], vintage_trim['t2'].shift(1)['1965-09-30':], c='purple', label='t = 3', alpha=1, s=20)
   ax2.plot(np.linspace(-0.025, 0.150), model0, color="red", lw=1.5)
   ax2.set_ylabel(r'$\pi_{t-1}$', fontsize=12)
   ax2.set_xlabel(r'$\pi_t$', fontsize=12)
   ax2.legend(['t = 3'], fontsize=12)

   ax3.scatter(vintage_trim['t3']['1965-09-30':], vintage_trim['t3'].shift(1)['1965-09-30':], c='black', label='t = 4', alpha=1, s=20)
   ax3.plot(np.linspace(-0.025, 0.150), model0, color="red", lw=1.5)
   ax3.set_xlabel(r'$\pi_t$', fontsize=12)
   ax3.legend(['t = 4'], fontsize=12)

   fig.legend(handles=ar)
   plt.tight_layout()
   plt.show()
   if save==True:
      fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)


def for_line(date, name=None, save=False, path='output/'):
   '''
   Plots annual CPI growth vintage vs forecast across time
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   mean_spf_trim = pd.read_csv('cleaned_data/mean_spf_trim.csv')
   vintage_trim = pd.read_csv('cleaned_data/vintage_trim.csv')

   vintage_trim['DATE'] = pd.to_datetime(vintage_trim['DATE'])
   mean_spf_trim['DATE'] = pd.to_datetime(mean_spf_trim['DATE'])
   vintage_trim = vintage_trim.loc[(vintage_trim['DATE'] >= '1981-09-01') & (vintage_trim['DATE'] <= date)]  # Filter data
   vintage_trim = vintage_trim.set_index('DATE')
   mean_spf_trim  = mean_spf_trim .loc[(mean_spf_trim ['DATE'] >= '1981-09-01') & (mean_spf_trim ['DATE'] <= str(pd.to_datetime(date) + pd.offsets.QuarterEnd(3)))]  # Filter data
   mean_spf_trim = mean_spf_trim.set_index('DATE')

   # Plotting
   fig, ax = plt.subplots(figsize=(12, 6))

   # Plot actuals and forecasts
   ax.plot(vintage_trim.index, vintage_trim['t3'], color="k", lw=1.5, label=r'$\pi_{t+3}$ Actuals')
   ax.plot(mean_spf_trim.index, mean_spf_trim['f_t3'].shift(-3), color='red', linestyle='dashed', lw=1.5, label=r'$f_{t-3}\pi_{t+3}$ Mean Forecasts')

   # Add recession shading
   ax1 = ax.twinx()
   ax1.fill_between(vintage_trim.index, 0, vintage_trim['rec'], color='grey', alpha=0.2)
   ax1.set_ylim(0, 1)
   ax1.get_yaxis().set_visible(False)

   # Legend and formatting
   fig.legend(loc='right', fontsize=12, bbox_to_anchor=(0.76, 0.75, 0.12, 0.25))
   plt.tight_layout()
   plt.show()
   if save==True:
      fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)


def est_plot(regs, regs_other, regs_color='black', regs_other_color='blue', name=None, save=False, path='output/'):
   '''
   Plots regression estimates w/ intervals
   regs_color: Estimates plot color
   regs_other_color: Comparative estimates plot color
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   fig, ax = plt.subplots(figsize=(9, 4))
   
   estimates = pd.DataFrame({'estimates': [regs[0][3].params[1], regs_other[0][3].params[1], regs[1][3].params[1], regs_other[1][3].params[1], regs[2][3].params[1], regs_other[2][3].params[1]]}, index=['(1)', '(2)', '(3)', '(4)',  '(5)',  '(6)'])
   errors = {'(1)': 1.96 * regs[0][3].bse[1], '(2)': 1.96 * regs_other[0][3].bse[1], '(3)': 1.96 * regs[1][3].bse[1], '(4)': 1.96 * regs_other[1][3].bse[1], '(5)': 1.96 * regs[2][3].bse[1], '(6)': 1.96 * regs_other[2][3].bse[1]}
   ax.axhline(0, ls='--', color='red')
   nr = ['(2)', '(4)', '(6)']
   width = 0.5  # Adjust the width of the error bars
   
   for idx, i in enumerate(estimates.index):
      x = idx + 1
      if i in nr:
         ax.errorbar(x - width/2, estimates.loc[i, 'estimates'], errors[i], capsize=3, fmt='o', color=f'{regs_other_color}')
      else:
         ax.errorbar(x + width/2, estimates.loc[i, 'estimates'], errors[i], capsize=3, fmt='o', color=f'{regs_color}')
    
   ax.set_xticks(range(1, len(estimates.index) + 1))
   ax.set_xticks([1.5, 3.5, 5.5])
   ax.set_xticklabels(['(1)', '(2)', '(3)'])
   
   plt.tight_layout()
   plt.show()
   if save == True:
    fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)


def est_plot_quad(regs, name=None, save=False, path='output/'):
   '''
   Plots all regression horizon estimates w/ intervals
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 4), sharex=True, sharey=True)
   handles = [ax1.errorbar([], [], [], fmt='o', color=color) for color in ['blue', 'purple', 'green', 'black']]
   labels = [r'$\pi_t$', r'$\pi_{t+1}$', r'$\pi_{t+2}$', r'$\pi_{t+3}$']
   fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.76, 0.73, 0.22, 0.25), ncol=2)

   estimates1 = pd.DataFrame({'estimates': [regs[0][0].params[1], regs[1][0].params[1], regs[2][0].params[1]]}, index=['(1)', '(2)', '(3)'])
   estimates2 = pd.DataFrame({'estimates': [regs[0][1].params[1], regs[1][1].params[1], regs[2][1].params[1]]}, index=['(1)', '(2)', '(3)'])
   estimates3 = pd.DataFrame({'estimates': [regs[0][2].params[1], regs[1][2].params[1], regs[2][2].params[1]]}, index=['(1)', '(2)', '(3)'])
   estimates4 = pd.DataFrame({'estimates': [regs[0][3].params[1], regs[1][3].params[1], regs[2][3].params[1]]}, index=['(1)', '(2)', '(3)'])

   errors1 = {'(1)': 1.96 * regs[0][0].bse[1], '(2)': 1.96 * regs[1][0].bse[1], '(3)': 1.96 * regs[2][0].bse[1]}
   errors2 = {'(1)': 1.96 * regs[0][1].bse[1], '(2)': 1.96 * regs[1][1].bse[1], '(3)': 1.96 * regs[2][1].bse[1]}
   errors3 = {'(1)': 1.96 * regs[0][2].bse[1], '(2)': 1.96 * regs[1][2].bse[1], '(3)': 1.96 * regs[2][2].bse[1]}
   errors4 = {'(1)': 1.96 * regs[0][3].bse[1], '(2)': 1.96 * regs[1][3].bse[1], '(3)': 1.96 * regs[2][3].bse[1]}

   for i in estimates1.index:
      ax1.errorbar(f'{i}', estimates1.loc[f'{i}', 'estimates'], errors1[i], capsize=3, fmt='o', color='blue')
      ax1.axhline(0, ls='--', color='red')
   for i in estimates2.index:
      ax2.errorbar(f'{i}', estimates2.loc[f'{i}', 'estimates'], errors2[i], capsize=3, fmt='o', color='purple')
      ax2.axhline(0, ls='--', color='red')
   for i in estimates3.index:
      ax3.errorbar(f'{i}', estimates3.loc[f'{i}', 'estimates'], errors3[i], capsize=3, fmt='o', color='green')
      ax3.axhline(0, ls='--', color='red')
   for i in estimates4.index:
      ax4.errorbar(f'{i}', estimates4.loc[f'{i}', 'estimates'], errors4[i], capsize=3, fmt='o', color='k')
      ax4.axhline(0, ls='--', color='red')
      
   plt.tight_layout()
   plt.show()
   if save == True:
    fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)
    

def params_plot(regs, parameter, regs_other, parameter_other, regs_color='black', regs_other_color='blue', name=None, save=False, path='output/'):
   '''
   Plots model paramters w/ intervals
   regs_color: Estimates plot color
   regs_other_color: Comparative estimates plot color
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   n = [(regs[0][3].params[1] + 1.96 * regs[0][3].bse[1]), (regs[1][3].params[1] + 1.96 * regs[1][3].bse[1]), (regs[2][3].params[1] + 1.96 * regs[2][3].bse[1]), (regs[3].params[1] + 1.96 * regs[3].bse[1])]
   nr = [(regs_other[0][3].params[1] + 1.96 * regs_other[0][3].bse[1]), (regs_other[1][3].params[1] + 1.96 * regs_other[1][3].bse[1]), (regs_other[2][3].params[1] + 1.96 * regs_other[2][3].bse[1]), (regs_other[3].params[1] + 1.96 * regs_other[3].bse[1])]

   a = estimations.params(n[0], n[1], n[2], n[3])
   b = estimations.params(nr[0], nr[1], nr[2], nr[3])
   errors = [(a[0]-parameter[0]), (b[0]-parameter_other[0]), (a[1]-parameter[1]), (b[1]-parameter_other[1]), (a[2]-parameter[2]), (b[2]-parameter_other[2]), (a[3]-parameter[3]), (b[3]-parameter_other[3])]
   
   fig, ax = plt.subplots(figsize=(9, 4))

   parameters = pd.DataFrame({'parameters': [parameter[0], parameter_other[0], parameter[1], parameter_other[1], parameter[2], parameter_other[2],parameter[3], parameter_other[3]]}, index=['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)'])

   errors = {'(1)': abs(errors[0]), '(2)': abs(errors[1]), '(3)': abs(errors[2]), '(4)': abs(errors[3]), '(5)': abs(errors[4]), '(6)': abs(errors[5]), '(7)': abs(errors[6]), '(8)': abs(errors[7])}

   nr = ['(2)', '(4)', '(6)', '(8)']
   width = 0.4  # Adjust the width of the error bars
   
   for idx, i in enumerate(parameters.index):
      x = idx + 1
      if i in nr:
         ax.errorbar(x - width/2, parameters.loc[i, 'parameters'], errors[i], capsize=3, fmt='o', color=f'{regs_other_color}')
      else:
         ax.errorbar(x + width/2, parameters.loc[i, 'parameters'], errors[i], capsize=3, fmt='o', color=f'{regs_color}')
    
   ax.set_xticks(range(1, len(parameters.index) + 1))
   ax.set_xticks([1.5, 3.5, 5.5, 7.5])
   ax.set_xticklabels([r'$\lambda$', r'$G$', r'$\theta^P$', r'$\theta^{FE}$'])
      
   plt.tight_layout()
   plt.show()
   if save == True:
    fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)
    

def params_plot_quad(regs, name=None, save=False, path='output/'):
   '''
   Plots all horizon model parameter estimates w/ intervals
   name: Name of file if saved
   save: Saving file to desired path
   path: Path to save
   '''
   nt0 = [(regs[0][0].params[1] + 1.96 * regs[0][0].bse[1]), (regs[1][0].params[1] + 1.96 * regs[1][0].bse[1]), (regs[2][0].params[1] + 1.96 * regs[2][0].bse[1]), (regs[3].params[1] + 1.96 * regs[3].bse[1])]
   nt1 = [(regs[0][1].params[1] + 1.96 * regs[0][1].bse[1]), (regs[1][1].params[1] + 1.96 * regs[1][1].bse[1]), (regs[2][1].params[1] + 1.96 * regs[2][1].bse[1]), (regs[3].params[1] + 1.96 * regs[3].bse[1])]
   nt2 = [(regs[0][2].params[1] + 1.96 * regs[0][2].bse[1]), (regs[1][2].params[1] + 1.96 * regs[1][2].bse[1]), (regs[2][3].params[1] + 1.96 * regs[2][2].bse[1]), (regs[3].params[1] + 1.96 * regs[3].bse[1])]
   nt3 = [(regs[0][3].params[1] + 1.96 * regs[0][3].bse[1]), (regs[1][3].params[1] + 1.96 * regs[1][3].bse[1]), (regs[2][3].params[1] + 1.96 * regs[2][3].bse[1]),(regs[3].params[1] + 1.96 * regs[3].bse[1])]

   a0 = estimations.params(nt0[0], nt0[1], nt0[2], nt0[3])
   a1 = estimations.params(nt1[0], nt1[1], nt1[2], nt1[3])
   a2 = estimations.params(nt2[0], nt2[1], nt2[2], nt2[3])
   a3 = estimations.params(nt3[0], nt3[1], nt3[2], nt3[3])
   
   parameter = [estimations.params(regs[0][0].params[1], regs[1][0].params[1], regs[2][0].params[1], regs[3].roots), estimations.params(regs[0][1].params[1], regs[1][1].params[1], regs[2][1].params[1], regs[3].roots), estimations.params(regs[0][2].params[1], regs[1][2].params[1], regs[2][2].params[1], regs[3].roots), estimations.params(regs[0][3].params[1], regs[1][3].params[1], regs[2][3].params[1], regs[3].roots)]
   
   errors0 = [(a0[0]-parameter[0][0]), (a0[1]-parameter[0][1]), (a0[2]-parameter[0][2]), (a0[3]-parameter[0][3])]
   errors1 = [(a1[0]-parameter[1][0]), (a1[1]-parameter[1][1]), (a1[2]-parameter[1][2]), (a1[3]-parameter[1][3])]
   errors2 = [(a2[0]-parameter[2][0]), (a2[1]-parameter[2][1]), (a2[2]-parameter[2][2]), (a2[3]-parameter[2][3])]
   errors3 = [(a3[0]-parameter[3][0]), (a3[1]-parameter[3][1]), (a3[2]-parameter[3][2]), (a3[3]-parameter[3][3])]
   
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), sharex=True)
   handles = [ax1.errorbar([], [], [], fmt='o', color=color) for color in ['blue', 'purple', 'green', 'black']]
   labels = [r'$\pi_t$', r'$\pi_{t+1}$', r'$\pi_{t+2}$', r'$\pi_{t+3}$']
   fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.76, 0.76, 0.23, 0.25), ncol=2)

   parameters0 = pd.DataFrame({'parameters': [parameter[0][0], parameter[0][1], parameter[0][2], parameter[0][3]]}, index=[r'$\lambda$', r'$G$', r'$\theta^P$', r'$\theta^{FE}$'])
   parameters1 = pd.DataFrame({'parameters': [parameter[1][0], parameter[1][1], parameter[1][2], parameter[1][3]]}, index=[r'$\lambda$', r'$G$', r'$\theta^P$', r'$\theta^{FE}$'])
   parameters2 = pd.DataFrame({'parameters': [parameter[2][0], parameter[2][1], parameter[2][2], parameter[2][3]]}, index=[r'$\lambda$', r'$G$', r'$\theta^P$', r'$\theta^{FE}$'])
   parameters3 = pd.DataFrame({'parameters': [parameter[3][0], parameter[3][1], parameter[3][2], parameter[3][3]]}, index=[r'$\lambda$', r'$G$', r'$\theta^P$', r'$\theta^{FE}$'])

   error0 = {r'$\lambda$': abs(errors0[0]), r'$G$': abs(errors0[1]), r'$\theta^P$': abs(errors0[2]), r'$\theta^{FE}$': abs(errors0[3])}
   error1 = {r'$\lambda$': abs(errors1[0]), r'$G$': abs(errors1[1]), r'$\theta^P$': abs(errors1[2]), r'$\theta^{FE}$': abs(errors1[3])}
   error2 = {r'$\lambda$': abs(errors2[0]), r'$G$': abs(errors2[1]), r'$\theta^P$': abs(errors2[2]), r'$\theta^{FE}$': abs(errors2[3])}
   error3 = {r'$\lambda$': abs(errors3[0]), r'$G$': abs(errors3[1]), r'$\theta^P$': abs(errors3[2]), r'$\theta^{FE}$': abs(errors3[3])}

   for i in parameters0.index:
      ax1.errorbar(f'{i}', parameters0.loc[f'{i}', 'parameters'], error0[i], capsize=3, fmt='o', color='blue')
   for i in parameters1.index:
      ax2.errorbar(f'{i}', parameters1.loc[f'{i}', 'parameters'], error1[i], capsize=3, fmt='o', color='purple')
   for i in parameters2.index:
      ax3.errorbar(f'{i}', parameters2.loc[f'{i}', 'parameters'], error2[i], capsize=3, fmt='o', color='green')
   for i in parameters3.index:
      ax4.errorbar(f'{i}', parameters3.loc[f'{i}', 'parameters'], error3[i], capsize=3, fmt='o', color='k')
      
   plt.tight_layout()
   plt.show()
   if save == True:
    fig.savefig(f'{path}{name}.png', bbox_inches='tight', dpi=200)
    

# rev_errs(regs=, date=, save=, name=)
# auto_reg(regs=, date=, save=, name=)
# for_line(date=, save=, name=)
# est_plot(regs=, regs_other=, save=, name=)
# params_plot(regs=, parameter=, regs_other=, parameter_other=, save=, name=)
# est_plot_quad(regs=, save=, name=
# est_plot_quad(regs=, save=, name=)
# params_plot_quad(regs=, save=, name=)