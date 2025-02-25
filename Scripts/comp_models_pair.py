"""Comparing the baseline and vae models for the values generated from external datasets(combinations) """

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# Load corr values from a CSV file
df_vae = pd.read_csv('/disk2/user/manche/Panacea/output/vae/corr_ed.csv')
#print(f"vae {df_vae.head()}")
df_397 = pd.read_csv('/disk2/user/manche/Panacea/output/model/corr_397.csv')
#print(f"397 {df_397.head()}")
df_428 = pd.read_csv('/disk2/user/manche/Panacea/output/model/corr_428.csv')
#print(f"428 {df_428.head()}")

#Assigning drug corr values from df vae to variables
vae_trametinib = df_vae['trametinib']
vae_palbociclib = df_vae['palbociclib']
vae_tp = df_vae['combo_tp']
vae_tamoxifen = df_vae['tamoxifen']
vae_mefloquine = df_vae['mefloquine']
vae_withaferin = df_vae['withaferin']
vae_tm = df_vae['combo_tm']
vae_tw = df_vae['combo_tw']
vae_mw = df_vae['combo_mw']

#Assigning drug corr values from df 397 to variables
model_trametinib = df_397['trametinib']
model_palbociclib = df_397['palbociclib']
model_tp = df_397['combo_tp']

#Assigning drug corr values from df 428 to variables
model_tamoxifen = df_428['tamoxifen']
model_mefloquine = df_428['mefloquine']
model_withaferin = df_428['withaferin']
model_tm = df_428['combo_tm']
model_tw = df_428['combo_tw']
model_mw = df_428['combo_mw']

# Define labels for each data point
labels = ['TP','TM', 'TW', 'MW' ]
colors = ['red', 'green', 'blue', 'orange']


# Define correlation values for two models
vae_corr = np.array([vae_tp, vae_tm, vae_tw, vae_mw])
model_corr = np.array([ model_tp, model_tm, model_tw, model_mw])
# Create scatter plot
plt.scatter(model_corr, vae_corr, c=colors)
for i, label in enumerate(labels):
    plt.annotate(label, (model_corr[i], vae_corr[i]), textcoords="offset points", xytext=(0,10), ha='center')
# Draw a diagonal line
#plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls='--', c='k')
# Set plot title and axis labels
plt.title('Comparison of Correlation Values between Baseline and VAE models')
plt.xlabel('Baseline Model Correlation Values')
plt.ylabel('VAE Correlation Values')

"""# Check for missing or constant values in arrays
missing_values = np.isnan(model_corr) | np.isnan(vae_corr)
constant_values = np.unique(model_corr).size == 1 or np.unique(vae_corr).size == 1
# If there are missing or constant values, remove them from arrays
if missing_values.any() or constant_values:
    model_corr = model_corr[~missing_values]
    vae_corr = vae_corr[~missing_values]
# Calculate correlation coefficient between the two sets of values
corr_coeff = np.corrcoef(model_corr, vae_corr, rowvar=False)[0,1]
# Calculate slope and intercept of correlation line
slope = corr_coeff * (np.std(vae_corr) / np.std(model_corr))
intercept = np.mean(vae_corr) - slope * np.mean(model_corr)
# Add correlation line to scatter plot
plt.plot(model_corr, slope * model_corr + intercept, c='black')"""

#plt.savefig('/disk2/user/manche/Panacea/output/plots/compare/compare_models.pdf', bbox_inches='tight')
plt.clf()


# Define labels for each data point
#labels = ['Trametinib', 'Palbociclib', 'Comb_TP', 'Tamoxifen', 'Mefloquine',
#          'Withaferin', 'Combo_TM', 'Combo_TW', 'Combo_MW' ]

# Create scatter plot
plt.scatter(['Baseline'] * len(model_corr), model_corr, c=colors)
plt.scatter(['VAE'] * len(vae_corr), vae_corr, c=colors)
#plt.scatter(['Baseline'] * len(model_corr), model_corr, label='Baseline', c=colors)
#plt.scatter(['VAE'] * len(vae_corr), vae_corr, label='VAE', c=colors)
# Add labels for each data point
for label, x, y in zip(labels, ['Baseline'] * len(model_corr), model_corr):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
for label, x, y in zip(labels, ['VAE'] * len(vae_corr), vae_corr):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
# Loop over correlation values and draw lines between similar points
for i in range(len(model_corr)):
    #if abs(model_corr[i] - vae_corr[i]) < 0.1:
    plt.plot(['Baseline', 'VAE'], [model_corr[i], vae_corr[i]], c=colors[i])
# Set plot title and axis labels
#plt.title('Evaluating the Performance of Baseline and VAE Models using Correlation Values')
plt.xlabel('Models')
plt.ylabel('Correlation Values')
# Add legend
#plt.legend()
# Set x-axis limits
plt.xlim([-1, 2])
plt.savefig('/disk2/user/manche/Panacea/output/plots/compare/compare_models1_pair.pdf', bbox_inches='tight')
plt.clf()

# Define colors
baseline_color = 'blue'
vae_color = 'orange'
# Create boxplots
data = [model_corr.ravel(), vae_corr.ravel()]
labels = ['Baseline Model', 'VAE Model']
boxprops = dict(linestyle='-', linewidth=2, color=baseline_color)
plt.boxplot(data[0], positions=[1], boxprops=boxprops)
boxprops = dict(linestyle='-', linewidth=2, color=vae_color)
plt.boxplot(data[1], positions=[2], boxprops=boxprops)
# Add axis labels and title
plt.xlabel('Model')
plt.ylabel('Correlation Values')
#plt.title('Evaluating the Performance of Baseline and VAE Models using Correlation Values')
# Add legend
plt.legend([patches.Patch(color=baseline_color), patches.Patch(color=vae_color)],
           ['Baseline Model', 'VAE Model'])
plt.title('Comparison of Correlation Values between Baseline and VAE Models')
#plt.savefig('/disk2/user/manche/Panacea/output/plots/compare/compare_models_box.pdf', bbox_inches='tight')
plt.clf()

# mean correlation values
baseline_mean = np.mean(model_corr)
vae_mean = np.mean(vae_corr)
print('Baseline model mean correlation:', baseline_mean)
print('VAE model mean correlation:', vae_mean)
