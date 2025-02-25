"""Script to validate the values of vae model with external datasets"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import vae_model as v
import seaborn as sns
sns.set(font_scale=1.2)
from tensorflow.keras.models import load_model
from scipy.stats import spearmanr

# Load the saved model
#combined_model = load_model('combined_model.h5')
#vae = load_model('vae.h5')

# Load the test dataset GSE110397
df1 = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE110397/GSE110397_entrez_imputed.csv')
df1 = df1.reset_index(drop=True)

# Loading pdata of test dataset
df1_pdata = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE110397/GSE110397_pdata.csv')
print(df1_pdata.head())
col_names = df1_pdata['short_name']
test_drug_names = ['Trametinib', 'Palbociclib']
# list of row names
#gene_names = df1.iloc[:, 0].tolist()
df1 = df1.T
print(df1.head())
df1 = df1.iloc[1:, 0:8113]

# Normalize the data using z-score normalization
#normalized_data1 = (df1 - df1.mean()) / df1.std()
# Define a function for z-score normalization
def zscore(row):
    numeric_values = pd.to_numeric(row, errors='coerce')
    return (numeric_values - numeric_values.mean()) / numeric_values.std()

# Apply z-score normalization to each column
normalized_data1 = df1.apply(zscore, axis=0)
print(f'norm data is {normalized_data1.head()}')

# Reorder the columns based on the original order
normalized_data1 = normalized_data1.loc[:, df1.columns]
# Replace all NaN values with 0
normalized_data1 = normalized_data1.fillna(0)
#Assigning normalised data to new variable
test_drug_data = normalized_data1
# setting drug names as index using col names
test_drug_data.index = col_names
#Calling trained models encoder to predict the single drug response and then compute the combined effect of two drugs
test_drug_latent = v.encoder.predict(test_drug_data)[2]
print(f"test drug latent is {test_drug_latent}")
# Convert the matrix to a DataFrame
test_drug_latent_df = pd.DataFrame(data=test_drug_latent, index=col_names)
print(f"test drug df is {test_drug_latent_df.head()}")

# Generate drug latent vectors
dmso_test = test_drug_latent_df.loc[['DMSO_1', 'DMSO_2']].mean()
test_drug_responses = []
for drug_name in test_drug_names:
    if drug_name == 'Trametinib':
        #test_drug_size = len(test_drug_latent_df.index.str.contains(drug_name))
        test_drug_latent_mean =  test_drug_latent_df.loc[['Trametinib_1', 'Trametinib_2']].mean()
        print(f"shape of test drug latent mean is {test_drug_latent_mean.shape}")
        test_drug_latent_mean = test_drug_latent_mean - dmso_test
        test_drug_responses.append((drug_name, test_drug_latent_mean))
    if drug_name == 'Palbociclib':
        test_drug_latent_mean =  test_drug_latent_df.loc[['Palbociclib_1', 'Palbociclib_2']].mean()
        test_drug_latent_mean = test_drug_latent_mean - dmso_test
        test_drug_responses.append((drug_name, test_drug_latent_mean))
#print(test_drug_responses[0][0])
#print(f"test drug response data {print(test_drug_responses[0][1])}")

# Generate combined drug responses
test_combined_responses = []
for i in range(len(test_drug_responses)):
    for j in range(i+1, len(test_drug_responses)):
        drug1_name, drug1_latent = test_drug_responses[i]
        drug2_name, drug2_latent = test_drug_responses[j]
        combined_latent = (drug1_latent + drug2_latent) / 2
        #combined_latent_inputs = np.concatenate((drug1_latent, drug2_latent), axis=None)
        combined_response = v.combined_model.predict(np.array([combined_latent]))
        test_combined_responses.append(((drug1_name, drug2_name), combined_response))
print(test_combined_responses)

#calculate the mean values for each drug and drug combo in the test dataset
print(f" norm data is {normalized_data1.head()}")
dmso = normalized_data1.loc[['DMSO_1', 'DMSO_2']].mean()
drug_Trametinib = normalized_data1.loc[['Trametinib_1', 'Trametinib_2']].mean()
norm_Trametinib = np.round(drug_Trametinib - dmso, 2)
print(f"norm_Trametinib is {norm_Trametinib}")
drug_Palbociclib = normalized_data1.loc[['Palbociclib_2', 'Palbociclib_2']].mean()
norm_Palbociclib = np.round(drug_Palbociclib - dmso, 2)
#print(f"norm_Palbociclib {norm_Palbociclib}")
combo = normalized_data1.loc[['combo_1', 'combo_2']].mean()
combo = np.round(combo - dmso, 2)
#print(f"combo {combo}")

#Evaluating the model by using  test drug data
test_loss = v.vae.evaluate(test_drug_data, batch_size=80)
print('Test loss of GSE110397:', test_loss)

#Use the pre-trained model to decode the single drug responses
#pred = vae.predict(test_drug_data)
test_drug_array = np.array([x[1] for x in test_drug_responses])
pred = v.decoder.predict(test_drug_array)
# Convert the matrix to a DataFrame
pred_df = pd.DataFrame(data=pred, columns=v.genes)
#pred_df = pred_df.T
print(pred_df.head())
Trametinib = pred_df.iloc[0]
pred_Trametinib = np.round(Trametinib, 2)
# Delete last two values that belongs to cell lines
#pred_Trametinib = pred_Trametinib[:-2]
print(f"pred trametinib after removing cell line columns {pred_Trametinib}")
Palbociclib = pred_df.iloc[1]
pred_Palbociclib = np.round(Palbociclib, 2)
#pred_Palbociclib = pred_Palbociclib[:-2]
drug_pair = test_combined_responses[0][0]
values = test_combined_responses[0][1]
# Convert to a pandas Series
values = pd.Series(values[0], dtype=np.float64)
#print(f"values {values}")
#print(f"pred _dmso {pred_dmso}")
pred_combo = np.round(values, 2)
#pred_combo = pred_combo[:-2]
print(f"pred combo {pred_combo}")

# Calculate the pearson correlation coefficient
corr_Trametinib = np.corrcoef(pred_Trametinib, norm_Trametinib)[0, 1]
corr_Palbociclib = np.corrcoef(pred_Palbociclib, norm_Palbociclib)[0, 1]
corr_combo = np.corrcoef(pred_combo, combo)[0, 1]

print(f"corr value for trametinib is {corr_Trametinib}")
print(f"corr value for Palbociclib is {corr_Palbociclib}")
print(f"corr value for combo is {corr_combo}")

# Calculate the Spearman correlation coefficient
corr_Trametinib_s, _ = spearmanr(pred_Trametinib, norm_Trametinib)
corr_Palbociclib_s, _ = spearmanr(pred_Palbociclib, norm_Palbociclib)
corr_combo_s, _ = spearmanr(pred_combo, combo)

print(f"corr value for trametinib s is {corr_Trametinib_s}")
print(f"corr value for Palbociclib s is {corr_Palbociclib_s}")
print(f"corr value for combo pt s is {corr_combo_s}")

# correlation matrix for Trametinib
data = np.stack([pred_Trametinib, norm_Trametinib, norm_Palbociclib], axis=1)
#corr_matrix = np.corrcoef(data, rowvar=False)
corr_matrix, _ = spearmanr(data, axis=0)
sns.heatmap(corr_matrix, annot=True, cmap='YlGn', xticklabels=['VAE_Trametinib','ED_Trametinib','ED_Palbociclib'],
           yticklabels=['VAE_Trametinib','ED_Trametinib','ED_Palbociclib'])
plt.title('Correlation Matrix for Trametinib')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_Trametinib.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for Palbociclib
data1 = np.stack([pred_Palbociclib, norm_Trametinib, norm_Palbociclib], axis=1)
#corr_matrix1 = np.corrcoef(data1, rowvar=False)
corr_matrix1, _ = spearmanr(data1, axis=0)
sns.heatmap(corr_matrix1, annot=True, cmap='YlGn', xticklabels=['VAE_Palbociclib','ED_Trametinib','ED_Palbociclib'],
           yticklabels=['VAE_Palbociclib','ED_Trametinib','ED_Palbociclib'])
plt.title('Correlation Matrix for Palbociclib')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_Palbociclib.pdf', bbox_inches='tight')
plt.clf()

# correlation matrix for combo pt
data2 = np.stack([norm_Palbociclib, norm_Trametinib, combo, pred_Palbociclib, pred_Trametinib, pred_combo], axis=1)
#corr_matrix2 = np.corrcoef(data2, rowvar=False)
corr_matrix2, _ = spearmanr(data2, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix2, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix2, annot=True, cmap='YlGn', mask=mask, xticklabels=['ED_Palbociclib','ED_Trametinib','ED_combo','VAE_Palbociclib','VAE_Trametinib','VAE_combo'],
           yticklabels=['ED_Palbociclib','ED_Trametinib','ED_combo','VAE_Palbociclib','VAE_Trametinib','VAE_combo'],
           linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination of pt')
# save the graph to pdf
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_combo_pt.pdf', bbox_inches='tight')
plt.clf()

##########################################################################################################################################

# Load the test dataset GSE149428
df2 = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE149428/GSE149428_entrez_imputed.csv')
df2 = df2.reset_index(drop=True)
# load pdata of external dataset
df2_pdata = pd.read_csv('/disk2/user/manche/Panacea/output/External_dataset/parsed/GSE149428/GSE149428_pdata.csv')
#print(df1_pdata.head())
col_names_1 = df2_pdata['short_name']
test_drug_names_1 = ['T_24', 'M_24', 'W_24']
df2 = df2.T
print(df2.head())
df2 = df2.iloc[1:, 0:8113]
# Normalize the data using z-score normalization
#normalized_data2 = (df2 - df2.mean()) / df2.std()
# Define a function for z-score normalization
def zscore(row):
    numeric_values = pd.to_numeric(row, errors='coerce')
    return (numeric_values - numeric_values.mean()) / numeric_values.std()

# Apply z-score normalization to each column
normalized_data2 = df2.apply(zscore, axis=0)
print(f'norm data is {normalized_data1.head()}')

# Reorder the columns based on the original order
normalized_data2 = normalized_data2.loc[:, df2.columns]
# Replace all NaN values with 0
normalized_data2 = normalized_data2.fillna(0)
test_drug_data_1 = normalized_data2
# adding two columns for the cell lines
#test_drug_data_1 = test_drug_data_1.assign(A549=0)
#test_drug_data_1 = test_drug_data_1.assign(MCF7=1)
test_drug_data_1.index = col_names_1
#Calling trained encoder to generate the drug latent variable for the ED
test_drug_latent_1 = v.encoder.predict(test_drug_data_1)[2]
# Convert the matrix to a DataFrame
test_drug_latent_1_df = pd.DataFrame(data=test_drug_latent_1, index=col_names_1)
# Generate drug latent vectors
dmso_test_1 = test_drug_latent_1_df.filter(regex='^DMSO_', axis=0).mean()
test_drug_responses_1 = []
for drug_name in test_drug_names_1:
    if drug_name == 'T_24':
        test_drug_latent_mean = test_drug_latent_1_df.filter(regex='^T_', axis=0).mean()
        test_drug_latent_mean = test_drug_latent_mean - dmso_test_1
        test_drug_responses_1.append((drug_name, test_drug_latent_mean))
    if drug_name == 'M_24':
        test_drug_latent_mean = test_drug_latent_1_df.filter(regex='^M_', axis=0).mean()
        test_drug_latent_mean = test_drug_latent_mean - dmso_test_1
        test_drug_responses_1.append((drug_name, test_drug_latent_mean))
    if drug_name == 'W_24':
        test_drug_latent_mean = test_drug_latent_1_df.filter(regex='^W_', axis=0).mean()
        test_drug_latent_mean = test_drug_latent_mean - dmso_test_1
        test_drug_responses_1.append((drug_name, test_drug_latent_mean))
print(test_drug_responses_1[0][0])
print(f"test drug response data {test_drug_responses_1[0][1]}")
print(test_drug_responses_1[1][0])
print(test_drug_responses_1[2][0])
# Generate combined drug responses for test dataset
test_combined_responses_1 = []
for i in range(len(test_drug_responses_1)):
    for j in range(i+1, len(test_drug_responses_1)):
        drug1_name, drug1_latent = test_drug_responses_1[i]
        drug2_name, drug2_latent = test_drug_responses_1[j]
        combined_latent_inputs = (drug1_latent + drug2_latent) / 2
        combined_response = v.combined_model.predict(np.array([combined_latent_inputs]))
        test_combined_responses_1.append(((drug1_name, drug2_name), combined_response))
print(test_combined_responses_1)

#calculate drug response in ED
dmso_1 = normalized_data2.filter(regex='^DMSO_', axis=0).mean()
#dmso_1 = normalized_data2.loc[['DMSO_24A','DMSO_24B','DMSO_24C']].mean()
drug_tamoxifen = normalized_data2.filter(regex='^T_', axis=0).mean()
#drug_tamoxifen = normalized_data2.loc[['T_24A','T_24B','T_24C']].mean()
norm_tamoxifen = np.round(drug_tamoxifen - dmso_1, 2)
print(f"norm_tamoxifen is {norm_tamoxifen}")
drug_mefloquine = normalized_data2.filter(regex='^M_', axis=0).mean()
#drug_mefloquine = normalized_data2.loc[['M_24A','M_24B','M_24C']].mean()
norm_mefloquine = np.round(drug_mefloquine - dmso_1 , 2)
drug_withaferin = normalized_data2.filter(regex='^W_', axis=0).mean()
#drug_withaferin = normalized_data2.loc[['W_24A','W_24B','W_24C']].mean()
norm_withaferin = np.round(drug_withaferin - dmso_1 , 2)
combo_tm = normalized_data2.filter(regex='^TM_', axis=0).mean()
#combo_tm = normalized_data2.loc[['TM_24A','TM_24B','TM_24C']].mean()
combo_tm = np.round(combo_tm - dmso_1, 2)
print(f"combo {combo_tm}")
#combo_tw = normalized_data2.filter(regex='^TW_', axis=0).mean()
combo_tw = normalized_data2.loc[['TW_24A','TW_24B','TW_24C']].mean()
combo_tw = np.round(combo_tw - dmso_1, 2)
#combo_mw = normalized_data2.filter(regex='^MW_', axis=0).mean()
combo_mw = normalized_data2.loc[['MW_24A','MW_24B','MW_24C']].mean()
combo_mw = np.round(combo_mw - dmso_1, 2)

#Predict the data by evaluating the model
test_loss_1 = v.vae.evaluate(test_drug_data_1, batch_size=80)
print('Test loss for ED GSE149428:', test_loss_1)

#Calling trained decoder to generate single drug responses
print(f"test drug response of GSE149428 {test_drug_responses_1}")
test_drug_array_1 = np.array([x[1] for x in test_drug_responses_1])
pred_1 = v.decoder.predict(test_drug_array_1)
# Convert the matrix to a DataFrame
pred_df_1 = pd.DataFrame(data=pred_1, columns=v.genes)
print(f" pred df 1 {pred_df_1.head()}")
tamoxifen = pred_df_1.iloc[0]
pred_tamoxifen = np.round(tamoxifen, 2)
# Delete last two values that belongs to cell lines
#pred_tamoxifen = pred_tamoxifen[:-2]
mefloquine = pred_df_1.iloc[1]
pred_mefloquine = np.round(mefloquine, 2)
#pred_mefloquine = pred_mefloquine[:-2]
withaferin = pred_df_1.iloc[2]
pred_withaferin = np.round(withaferin, 2)
#pred_withaferin = pred_withaferin[:-2]
drug_pair_1 = test_combined_responses_1[0][0]
values_1 = test_combined_responses_1[0][1]
print(f"values1 is {values_1}")
drug_pair_2 = test_combined_responses_1[1][0]
values_2 = test_combined_responses_1[1][1]
drug_pair_3 = test_combined_responses_1[2][0]
values_3 = test_combined_responses_1[2][1]
# Convert to a pandas Series
values_1 = pd.Series(values_1[0], dtype=np.float64)
print(f"values1 after converting to pd series {values_1}")
values_2 = pd.Series(values_2[0], dtype=np.float64)
values_3 = pd.Series(values_3[0], dtype=np.float64)
print(f"drug pair 1 is {drug_pair_1}")
print(f"drug pair 2 is {drug_pair_2}")
print(f"drug pair 3 is {drug_pair_3}")
pred_combo_tm = np.round(values_1, 2)
#pred_combo_tm = pred_combo_tm[:-2]
pred_combo_tw = np.round(values_2, 2)
#pred_combo_tw = pred_combo_tw[:-2]
pred_combo_mw = np.round(values_3, 2)
#pred_combo_mw =pred_combo_mw[:-2]

# Calculate the pearson correlation coefficient
corr_tamoxifen = np.corrcoef(pred_tamoxifen, norm_tamoxifen)[0, 1]
corr_mefloquine = np.corrcoef(pred_mefloquine, norm_mefloquine)[0, 1]
corr_withaferin = np.corrcoef(pred_withaferin, norm_withaferin)[0, 1]
corr_combo_tm = np.corrcoef(pred_combo_tm, combo_tm)[0, 1]
corr_combo_tw = np.corrcoef(pred_combo_tw, combo_tw)[0, 1]
corr_combo_mw = np.corrcoef(pred_combo_mw, combo_mw)[0, 1]

print(f"corr value for tamoxifen is {corr_tamoxifen}")
print(f"corr value for mefloquine is {corr_mefloquine}")
print(f"corr value for withaferin is {corr_withaferin}")
print(f"corr value for combo tm is {corr_combo_tm}")
print(f"corr value for combo tw is {corr_combo_tw}")
print(f"corr value for combo mw is {corr_combo_mw}")

# Calculate the Spearman correlation coefficient
corr_tamoxifen_s, _ = spearmanr(pred_tamoxifen, norm_tamoxifen)
print(f"shape of cmap tam value is {pred_tamoxifen.shape}")
print(f"shape of norm tam value is {norm_tamoxifen.shape}")
corr_mefloquine_s, _ = spearmanr(pred_mefloquine, norm_mefloquine)
corr_withaferin_s, _ = spearmanr(pred_withaferin, norm_withaferin)

corr_combo_tm_s, _ = spearmanr(pred_combo_tm, combo_tm)
corr_combo_tw_s, _ = spearmanr(pred_combo_tw, combo_tw)
corr_combo_mw_s, _ = spearmanr(pred_combo_mw, combo_mw)

print(f"corr value for tamoxifen s is {corr_tamoxifen_s}")
print(f"corr value for mefloquine s is {corr_mefloquine_s}")
print(f"corr value for withaferin s is {corr_withaferin_s}")
print(f"corr value for combo tm s is {corr_combo_tm_s}")
print(f"corr value for combo tw s is {corr_combo_tw_s}")
print(f"corr value for combo mw s is {corr_combo_mw_s}")

# Test
test_1, _ = spearmanr(norm_mefloquine, norm_tamoxifen)
test_2, _ = spearmanr(norm_withaferin, norm_tamoxifen)
test_3, _ = spearmanr(combo_tm, norm_tamoxifen)
print(f"corr mef with tam {test_1}")
print(f'corr with with tam {test_2}')
print(f'corr tm with tam {test_3}')

# Correlation matrix for drug tamoxifen
data3 = np.stack([pred_tamoxifen, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix3 = np.corrcoef(data3, rowvar=False)
corr_matrix3, _ = spearmanr(data3, axis=0)
sns.heatmap(corr_matrix3, annot=True, cmap='YlGn', xticklabels=['VAE_tamoxifen','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['VAE_tamoxifen','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for tamoxifen')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_tamoxifen.pdf', bbox_inches='tight')
plt.clf()

# Correlation matrix for drug mefloquine
data4 = np.stack([pred_mefloquine, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix4 = np.corrcoef(data4, rowvar=False)
corr_matrix4, _ = spearmanr(data4, axis=0)
sns.heatmap(corr_matrix4, annot=True, cmap='YlGn', xticklabels=['VAE_mefloquine','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['VAE_mefloquine','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for mefloquine')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_mefloquine.pdf', bbox_inches='tight')
plt.clf()

# Correlation matrix for drug withaferin
data5 = np.stack([pred_withaferin, norm_tamoxifen, norm_mefloquine, norm_withaferin], axis=1)
#corr_matrix5 = np.corrcoef(data5, rowvar=False)
corr_matrix5, _ = spearmanr(data5, axis=0)
sns.heatmap(corr_matrix5, annot=True, cmap='YlGn', xticklabels=['VAE_withaferin','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'],
           yticklabels=['VAE_withaferin','ED_tamoxifen','ED_mefloquine', 'ED_withaferin'])
plt.title('Correlation Matrix for withaferin')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_withaferin.pdf', bbox_inches='tight')
plt.clf()

# Correlation matrix for combo tm
data6 = np.stack([norm_tamoxifen, norm_mefloquine, combo_tm, pred_tamoxifen, pred_mefloquine, pred_combo_tm], axis=1)
#corr_matrix6 = np.corrcoef(data6, rowvar=False)
corr_matrix6, _ = spearmanr(data6, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix6, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix6, annot=True, cmap='YlGn',mask=mask, xticklabels=['ED_tamoxifen', 'ED_mefloquine','ED_combo','VAE_tamoxifen','VAE_mefloquine','VAE_combo'],
           yticklabels=['ED_tamoxifen', 'ED_mefloquine','ED_combo','VAE_tamoxifen','VAE_mefloquine','VAE_combo'],
           linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination tm')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_combo_tm.pdf', bbox_inches='tight')
plt.clf()

#Correlation matrix for combo tw
data7 = np.stack([norm_tamoxifen, norm_withaferin, combo_tw, pred_tamoxifen, pred_withaferin, pred_combo_tw], axis=1)
#corr_matrix7 = np.corrcoef(data7, rowvar=False)
corr_matrix7, _ = spearmanr(data7, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix7, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix7, annot=True, cmap='YlGn',mask=mask, xticklabels=['ED_tamoxifen', 'ED_withaferin','ED_combo','VAE_tamoxifen', 'VAE_withaferin','VAE_combo'],
           yticklabels=['ED_tamoxifen', 'ED_withaferin','ED_combo','VAE_tamoxifen', 'VAE_withaferin','VAE_combo'],
           linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination tw')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_combo_tw.pdf', bbox_inches='tight')
plt.clf()

# Correlation matrix for combo mw
data8 = np.stack([norm_withaferin, norm_mefloquine, combo_mw, pred_withaferin, pred_mefloquine,pred_combo_mw ], axis=1)
#corr_matrix8 = np.corrcoef(data8, rowvar=False)
corr_matrix8, _ = spearmanr(data8, axis=0)
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix8, dtype=bool))
mask[np.diag_indices_from(mask)] = False
# Set the background color to white
sns.set_style("white")
# Plot the triangular heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix8, annot=True, cmap='YlGn',mask=mask, xticklabels=['ED_withaferin', 'ED_mefloquine','ED_combo','VAE_withaferin', 'VAE_mefloquine','VAE_combo'],
           yticklabels=['ED_withaferin', 'ED_mefloquine','ED_combo','VAE_withaferin', 'VAE_mefloquine','VAE_combo'],
           linewidths=0, vmin=0, vmax=1, cbar=True)
#plt.title('Correlation Matrix for combination mw')
plt.savefig('/disk2/user/manche/Panacea/output/plots/vae/corr_matrix_combo_mw.pdf', bbox_inches='tight')
plt.clf()

#Create a dictionary to store corr values of drugs and drug combinations
corr_dict = {'tamoxifen': [corr_tamoxifen_s], 'mefloquine': [corr_mefloquine_s],
        'withaferin': [corr_withaferin_s], 'combo_tm': [corr_combo_tm_s],
        'combo_tw': [corr_combo_tw_s], 'combo_mw':[corr_combo_mw_s],
        'trametinib':[corr_Trametinib_s], 'palbociclib':[corr_Palbociclib_s],
        'combo_tp':[corr_combo_s]}
# Create a DataFrame from the dictionary
corr_dict_df = pd.DataFrame(corr_dict)
# Write the DataFrame to a CSV file
corr_dict_df.to_csv('/disk2/user/manche/Panacea/output/vae/corr_ed.csv', index=False)

