"""Evaluating baseline model values with external dataset GSE110397"""

#Import libraries
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import re
from scipy.stats import spearmanr
warnings.filterwarnings("ignore", message="The default value of numeric_only in DataFrame.mean is deprecated.")
sns.set(font_scale=1.2)

# Load gene expression data from a cmap tsv file which is rank normalized
data = pd.read_csv('/disk2/user/manche/Panacea/output/cmap/merged_with_ctl_138drugs_before_ranked.tsv')
print(data.head())
data = data.iloc[:, 1:]


# Load gene expression data from external datasets
df2 = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE110397/GSE110397_entrez_ranked.csv')
df2 = df2.reset_index(drop=True)
normalized_data1 = df2

drug_list = ['trametinib','palbociclib','DMSO']
#merged_df = pd.concat([normalized_data, temp], axis=1)
#print(f"merged df drugs are {merged_df['cmap_name'].unique()}")
merged_df = data
print(f"merged df is {merged_df}")
merged_df = merged_df[merged_df['cmap_name'].isin(drug_list)]
merged_df = merged_df.loc[(merged_df['cell_line'] == 'A549') ,:]
"""merged_df['time'] = merged_df['time'].astype(str)
merged_df['dose'] = merged_df['dose'].astype(str)
print(f"len of merged df is {len(merged_df)}")"""
print(f"merged df is {merged_df}")

normalized_dict = {}
# select unique plates and drugs
plates = merged_df['rna_plate'].unique()
drugs = merged_df['cmap_name'].unique()
print(f"drugs are {drugs}")
#initialize variables
normalized_value = np.zeros(8113)
# defining function
def process_plate(drug, plate, merged_df):
    # skip the DMSO drug
    if drug == 'DMSO':
        return [] #temp
    dmso_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == 'DMSO'), :].mean().iloc[:-1]
    #dmso_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == 'DMSO'), :].iloc[:, :8113].mean()
    #print(f"dmso is {dmso_value}")
    if dmso_value.isna().any().any():
        return [] #temp
    drug_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == drug), :].mean() #.iloc[:-1]
    #print(f"drug is {drug_value}")
    if drug_value.isna().any().any():
        return [] #temp
    normalized_value = np.round(drug_value - dmso_value, 2)
    return normalized_value.tolist()

num_cores = 16

for drug in drugs:
    # skip the DMSO drug
    if drug == 'DMSO':
        continue
    print(f"drug is {drug}")
    temp = []
    results = Parallel(n_jobs=num_cores)(
        delayed(process_plate)(drug, plate, merged_df) for plate in plates
    )
    #removes empty list from the results list
    results = [r for r in results if r]
    for result in results:
        temp += result
        temp = np.array(temp)
        iter = results.index(result) + 1
        #temp = temp / (results.index(result) + 1)
        print(f"length of temp+ is {len(temp)}")
    temp = np.array(temp)
    print(f"iter value is {iter}")
    temp = temp / iter
    normalized_dict[drug] = temp
print(f"dict  is {normalized_dict}")

# Reading external dataset GSE110397
dmso_GSE110397 = np.mean(normalized_data1[['DMSO_1', 'DMSO_2']], axis=1)
drug_Trametinib = np.mean(normalized_data1[['Trametinib_1', 'Trametinib_2']], axis=1)
norm_Trametinib = np.round(drug_Trametinib - dmso_GSE110397, 2)
print(f"norm_Trametinib after applying log is {norm_Trametinib}")
drug_Palbociclib = np.mean(normalized_data1[['Palbociclib_1', 'Palbociclib_2']], axis=1)
norm_Palbociclib = np.round(drug_Palbociclib - dmso_GSE110397, 2)
print(f"norm_Palbociclib {norm_Palbociclib}")

drug_pt = np.mean(normalized_data1[['combo_1', 'combo_2']], axis=1)
norm_pt = np.round(drug_pt - dmso_GSE110397, 2)
print(f"norm pt is {norm_pt}")

#comparing cmap with external dataset GSE110397
cmap_Trametinib = normalized_dict['trametinib']
cmap_Trametinib = cmap_Trametinib.ravel()
print(f"cmap_Trametinib value is {cmap_Trametinib}")
cmap_Palbociclib = normalized_dict['palbociclib']
cmap_Palbociclib = cmap_Palbociclib.ravel()

# calculating the meAN OF PAir od combinations
cmap_pt = (cmap_Trametinib + cmap_Palbociclib) / 2

# Calculate the pearson correlation coefficient
corr_Trametinib = np.corrcoef(cmap_Trametinib, norm_Trametinib)[0, 1]
corr_Palbociclib = np.corrcoef(cmap_Palbociclib, norm_Palbociclib)[0, 1]
corr_pt = np.corrcoef(cmap_pt, norm_pt)[0, 1]

print(f"corr value for trametinib is {corr_Trametinib}")
print(f"corr value for Palbociclib is {corr_Palbociclib}")
print(f"corr value for combo pt is {corr_pt}")

# Calculate the Spearman correlation coefficient
corr_Trametinib_s, _ = spearmanr(cmap_Trametinib, norm_Trametinib)
corr_Palbociclib_s, _ = spearmanr(cmap_Palbociclib, norm_Palbociclib)
corr_pt_s, _ = spearmanr(cmap_pt, norm_pt)

print(f"corr value for trametinib s is {corr_Trametinib_s}")
print(f"corr value for Palbociclib s is {corr_Palbociclib_s}")
print(f"corr value for combo pt s is {corr_pt_s}")

# correlation matrix for drug Trametinib
data = np.stack([cmap_Trametinib, norm_Trametinib, norm_Palbociclib], axis=1)
#corr_matrix = np.corrcoef(data, rowvar=False)
corr_matrix, _ = spearmanr(data, axis=0)
sns.heatmap(corr_matrix, annot=True, cmap='YlGn', xticklabels=['BSL_Trametinib','ED_Trametinib','ED_Palbociclib'],
           yticklabels=['BSL_Trametinib','ED_Trametinib','ED_Palbociclib'])
plt.title('Correlation Matrix for Trametinib')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_Trametinib.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for drug Palbociclib
data1 = np.stack([cmap_Palbociclib, norm_Trametinib, norm_Palbociclib], axis=1)
#corr_matrix1 = np.corrcoef(data1, rowvar=False)
corr_matrix1, _ = spearmanr(data1, axis=0)
sns.heatmap(corr_matrix1, annot=True, cmap='YlGn', xticklabels=['BSL_Palbociclib','ED_Trametinib','ED_Palbociclib'],
           yticklabels=['BSL_Palbociclib','ED_Trametinib','ED_Palbociclib'])
plt.title('Correlation Matrix for Palbociclib')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_Palbociclib.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for combo
data2 = np.stack([norm_Palbociclib,norm_Trametinib,norm_pt,cmap_Palbociclib,cmap_Trametinib,cmap_pt], axis=1)
#corr_matrix2 = np.corrcoef(data2, rowvar=False)
corr_matrix2, _ = spearmanr(data2, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix2, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix2, annot=True, cmap="YlGn" ,mask=mask, xticklabels=['ED_Palbociclib', 'ED_Trametinib','ED_combo','BSL_Palbociclib', 'BSL_Trametinib','BSL_combo'],
           yticklabels=['ED_Palbociclib','ED_Trametinib','ED_combo','BSL_Palbociclib','BSL_Trametinib','BSL_combo'], square=True,
           linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination pt')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_combo_pt.pdf', bbox_inches='tight')


#Create a dictionary to store corr values of drugs and drug combinations
corr_dict = {'trametinib':[corr_Trametinib_s], 'palbociclib':[corr_Palbociclib_s],
        'combo_tp':[corr_pt_s]}
# Create a DataFrame from the dictionary
corr_dict_df = pd.DataFrame(corr_dict)
# Write the DataFrame to a CSV file
corr_dict_df.to_csv('/disk2/user/manche/Panacea/output/model/corr_397.csv', index=False)


