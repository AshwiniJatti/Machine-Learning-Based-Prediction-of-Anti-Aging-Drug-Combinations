"""Evaluating baseline model values with external dataset GSE149428"""

#Import libraries
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore", message="The default value of numeric_only in DataFrame.mean is deprecated.")
sns.set(font_scale=1.2)

# Load gene expression data from a cmap TSV file which is rank normalized
data = pd.read_csv('/disk2/user/manche/Panacea/output/cmap/merged_with_ctl_138drugs_before_ranked.tsv')
data = data.iloc[:, 1:]
merged_df = data

# Load gene expression data from external datasets
df3 = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE149428/GSE149428_entrez_ranked.csv')
df3 = df3.reset_index(drop=True)
normalized_data1 = df3
drug_list = ['tamoxifen','mefloquine','withaferin-a', 'DMSO']
#merged_df = pd.concat([normalized_data, temp], axis=1)
#print(f"merged df is {merged_df.head()}")
merged_df = merged_df[merged_df['cmap_name'].isin(drug_list)]
merged_df = merged_df.loc[(merged_df['cell_line'] == 'MCF7'),:]

normalized_dict = {}
# select unique plates and drugs
plates = merged_df['rna_plate'].unique()
drugs = merged_df['cmap_name'].unique()
#initialize variables
normalized_value = np.zeros(8113)

# defining function to calculate the difference between dmso and drug value across all the plates
def process_plate(drug, plate, merged_df):
    # skip the DMSO drug
    if drug == 'DMSO':
        return [] #temp
    dmso_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == 'DMSO'), :].mean().iloc[:-1]
    if dmso_value.isna().any().any():
        return [] #temp
    drug_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == drug), :].mean() #.iloc[1:]
    if drug_value.isna().any().any():
        return [] #temp
    normalized_value = np.round(drug_value - dmso_value, 2)
    return normalized_value.tolist()

#setting number of cores for parallel computing
num_cores = 12

#computing drug responses for each drug present in the drug list
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
        #print(f"length of temp+ is {len(temp)}")
    print(f"length of temp+ is {len(temp)}")
    #temp = np.array(temp)
    print(f"iter value is {iter}")
    temp = temp / iter
    print(len(temp))
    normalized_dict[drug] = temp
print(f"dict  is {normalized_dict}")

# Reading external dataset GSE149428
print(f"norm data1 is {normalized_data1.head()}")
#dmso_GSE149428 = np.mean(normalized_data1[['DMSO_24A','DMSO_24B','DMSO_24C']], axis=1)
dmso_GSE149428 = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('DMSO_')]].mean(axis=1)
#dmso_GSE149428 = normalized_data1.loc[:, normalized_data1.columns.str.startswith('DMSO_')].mean()
print(f"dmso of ed is {dmso_GSE149428}")
#drug_tamoxifen = np.mean(normalized_data1[['T_24A','T_24B','T_24C']], axis=1)
drug_tamoxifen = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('T_')]].mean(axis=1)
norm_tamoxifen = np.round(drug_tamoxifen - dmso_GSE149428 , 2)
print(f"norm tam value is {norm_tamoxifen}")
#drug_mefloquine = np.mean(normalized_data1[['M_24A','M_24B','M_24C']], axis=1)
drug_mefloquine = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('M_')]].mean(axis=1)
norm_mefloquine = np.round(drug_mefloquine - dmso_GSE149428 , 2)
#drug_withaferin = np.mean(normalized_data1[['W_24A','W_24B','W_24C']], axis=1)
drug_withaferin = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('W_')]].mean(axis=1)
norm_withaferin = np.round(drug_withaferin - dmso_GSE149428 , 2)

#drug_tm = np.mean(normalized_data1[['TM_24A','TM_24B','TM_24C']], axis=1)
drug_tm = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('TM_')]].mean(axis=1)
norm_tm = np.round(drug_tm - dmso_GSE149428 , 2)

#drug_tw = np.mean(normalized_data1[['TW_24A','TW_24B','TW_24C']], axis=1)
drug_tw = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('TW_')]].mean(axis=1)
norm_tw = np.round(drug_tw - dmso_GSE149428 , 2)

#drug_mw = np.mean(normalized_data1[['MW_24A','MW_24B','MW_24C']], axis=1)
drug_mw = normalized_data1.loc[:, normalized_data1.columns[normalized_data1.columns.str.startswith('MW_')]].mean(axis=1)
norm_mw = np.round(drug_mw - dmso_GSE149428 , 2)

#comparing cmap with external dataset GSE149428
cmap_tamoxifen = normalized_dict['tamoxifen']
cmap_tamoxifen = cmap_tamoxifen.ravel()
print(f"cmap_tamoxifen value is {cmap_tamoxifen}")
print(type(cmap_tamoxifen))
cmap_mefloquine = normalized_dict['mefloquine']
#cmap_mefloquine = cmap_mefloquine.ravel()
cmap_withaferin = normalized_dict['withaferin-a']
#cmap_withaferin = cmap_withaferin.ravel()

#calculate mean values for the drug pairs
cmap_tm = (cmap_tamoxifen + cmap_mefloquine) / 2
print(f"cmap tm value is {cmap_tm}")

cmap_tw = (cmap_tamoxifen + cmap_withaferin) / 2
print(f"cmap tw value is {cmap_tw}")

cmap_mw = (cmap_mefloquine + cmap_withaferin) / 2
print(f"cmap mw value is {cmap_mw}")

# Calculate the pearson correlation coefficient
corr_tamoxifen = np.corrcoef(cmap_tamoxifen, norm_tamoxifen)[0, 1]
print(f"shape of cmap tam value is {cmap_tamoxifen.shape}")
print(f"shape of norm tam value is {norm_tamoxifen.shape}")
corr_mefloquine = np.corrcoef(cmap_mefloquine, norm_mefloquine)[0, 1]
corr_withaferin = np.corrcoef(cmap_withaferin, norm_withaferin)[0, 1]

corr_tm = np.corrcoef(cmap_tm, norm_tm)[0, 1]
corr_tw = np.corrcoef(cmap_tw, norm_tw)[0, 1]
corr_mw = np.corrcoef(cmap_mw, norm_mw)[0, 1]

print(f"corr value for tamoxifen is {corr_tamoxifen}")
print(f"corr value for mefloquine is {corr_mefloquine}")
print(f"corr value for withaferin is {corr_withaferin}")
print(f"corr value for combo tm is {corr_tm}")
print(f"corr value for combo tw is {corr_tw}")
print(f"corr value for combo mw is {corr_mw}")

# Calculate the Spearman correlation coefficient
corr_tamoxifen_s, _ = spearmanr(cmap_tamoxifen, norm_tamoxifen)
print(f"shape of cmap tam value is {cmap_tamoxifen.shape}")
print(f"shape of norm tam value is {norm_tamoxifen.shape}")
corr_mefloquine_s, _ = spearmanr(cmap_mefloquine, norm_mefloquine)
corr_withaferin_s, _ = spearmanr(cmap_withaferin, norm_withaferin)

corr_tm_s, _ = spearmanr(cmap_tm, norm_tm)
corr_tw_s, _ = spearmanr(cmap_tw, norm_tw)
corr_mw_s, _ = spearmanr(cmap_mw, norm_mw)

print(f"corr value for tamoxifen s is {corr_tamoxifen_s}")
print(f"corr value for mefloquine s is {corr_mefloquine_s}")
print(f"corr value for withaferin s is {corr_withaferin_s}")
print(f"corr value for combo tm s is {corr_tm_s}")
print(f"corr value for combo tw s is {corr_tw_s}")
print(f"corr value for combo mw s is {corr_mw_s}")

# Test
test_1, _ = spearmanr(norm_mefloquine, norm_tamoxifen)
test_2, _ = spearmanr(norm_withaferin, norm_tamoxifen)
test_3, _ = spearmanr(norm_tm, norm_tamoxifen)
print(f"corr mef with tam {test_1}")
print(f'corr with with tam {test_2}')
print(f'corr tm with tam {test_3}')

# correlation matrix for  drug tamoxifen
data = np.stack([cmap_tamoxifen, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix = np.corrcoef(data, rowvar=False)
corr_matrix, _ = spearmanr(data, axis=0)
sns.heatmap(corr_matrix, annot=True, cmap='YlGn', xticklabels=['BSL_tamoxifen','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['BSL_tamoxifen','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for tamoxifen')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_tamoxifen.pdf', bbox_inches='tight')
plt.clf()

#correlation matrix for for drug mefloquine
data1 = np.stack([cmap_mefloquine, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix1 = np.corrcoef(data1, rowvar=False)
corr_matrix1, _ = spearmanr(data1, axis=0)
sns.heatmap(corr_matrix1, annot=True, cmap='YlGn', xticklabels=['BSL_mefloquine','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['BSL_mefloquine','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for mefloquine')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_mefloquine.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for drug withaferin
data2 = np.stack([cmap_withaferin, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix2 = np.corrcoef(data2, rowvar=False)
corr_matrix2, _ = spearmanr(data2, axis=0)
sns.heatmap(corr_matrix2, annot=True, cmap='YlGn', xticklabels=['BSL_withaferin','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['BSL_withaferin','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for withaferin')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_withaferin.pdf', bbox_inches='tight')
plt.clf()

#correlation matrix for combo  tm
data3 = np.stack([norm_tamoxifen, norm_mefloquine, norm_tm, cmap_tamoxifen, cmap_mefloquine, cmap_tm], axis=1)
#corr_matrix3 = np.corrcoef(data3, rowvar=False)
corr_matrix3, _ = spearmanr(data3, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix3, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix3, annot=True, cmap='YlGn', mask = mask, xticklabels=['ED_tamoxifen', 'ED_mefloquine','ED_combo','BSL_tamoxifen', 'BSL_mefloquine', 'BSL_combo'],
           yticklabels=['ED_tamoxifen', 'ED_mefloquine','ED_combo','BSL_tamoxifen', 'BSL_mefloquine', 'BSL_combo'],
           square=True, linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination tm')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_combo_tm.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for combo tw
data4 = np.stack([norm_tamoxifen, norm_withaferin, norm_tw, cmap_tamoxifen, cmap_withaferin, cmap_tw], axis=1)
#corr_matrix4 = np.corrcoef(data4, rowvar=False)
corr_matrix4, _ = spearmanr(data4, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix4, dtype=bool))
mask[np.diag_indices_from(mask)] = False
sns.set_style("white")
# Plot the triangular heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix4, annot=True, cmap='YlGn', mask=mask, xticklabels=['ED_tamoxifen', 'ED_withaferin','ED_combo', 'BSL_tamoxifen', 'BSL_withaferin', 'BSL_combo'],
           yticklabels=['ED_tamoxifen', 'ED_withaferin','ED_combo', 'BSL_tamoxifen', 'BSL_withaferin', 'BSL_combo'],
           square=True, linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination tw')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_combo_tw.pdf', bbox_inches='tight')
plt.clf()

#correlation matrix for for combo mw
data5 = np.stack([norm_withaferin, norm_mefloquine, norm_mw, cmap_withaferin, cmap_mefloquine, cmap_mw], axis=1)
#corr_matrix5 = np.corrcoef(data5, rowvar=False)
corr_matrix5, _ = spearmanr(data5, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix5, dtype=bool))
mask[np.diag_indices_from(mask)] = False
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix5, annot=True, cmap='YlGn', mask=mask, xticklabels=['ED_withaferin', 'ED_mefloquine','ED_combo', 'BSL_withaferin', 'BSL_mefloquine', 'BSL_combo'],
           yticklabels=['ED_withaferin', 'ED_mefloquine','ED_combo', 'BSL_withaferin', 'BSL_mefloquine', 'BSL_combo'],
           square=True, linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination mw')
plt.savefig('/disk2/user/manche/Panacea/output/plots/model/corr_matrix_combo_mw.pdf', bbox_inches='tight')
plt.clf()

#Create a dictionary to store corr values of drugs and drug combinations
corr_dict = {'tamoxifen': [corr_tamoxifen_s], 'mefloquine': [corr_mefloquine_s],
        'withaferin': [corr_withaferin_s], 'combo_tm': [corr_tm_s],
        'combo_tw': [corr_tw_s], 'combo_mw':[corr_mw_s]}
# Create a DataFrame from the dictionary
corr_dict_df = pd.DataFrame(corr_dict)
# Write the DataFrame to a CSV file
corr_dict_df.to_csv('/disk2/user/manche/Panacea/output/model/corr_428.csv', index=False)
