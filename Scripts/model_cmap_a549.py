"""Baseline model to generate single and combined drug responses for the cell line a549.
Generated responses were stored in the csv files """

#Importing libraries
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load gene expression data from a TSV file
data = pd.read_csv('/disk2/user/manche/Panacea/output/cmap/merged_with_ctl_134drugs_before_ranked.tsv')
print(data['cell_line'].unique())
data = data.iloc[:, 1:]

merged_df = data
merged_df = merged_df.loc[(merged_df['cell_line'] == 'A549'),:]
print(merged_df.head())
column_names = list(merged_df.columns[0:8113])
column_names.insert(0, 'Drug_name')
print(f"col names are {column_names[0:5]}")
print(f"len of col names is {len(column_names)}")
#initialize an empty dictionary to store the normalized values
normalized_dict = {}
# select unique plates and drugs
plates = merged_df['rna_plate'].unique()
drugs = merged_df['cmap_name'].unique()
#initialize variables
normalized_value = np.zeros(8113)

# defining function to compute the difference between dmso and drug across all the plates
def process_plate(drug, plate, merged_df):
    # skip the DMSO drug
    if drug == 'DMSO':
        return []
    dmso_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == 'DMSO'), :].mean().iloc[:-1]
    #print(f"dmso value is {dmso_value}")
    if dmso_value.isna().any().any():
        return []
    drug_value = merged_df.loc[(merged_df['rna_plate'] == plate) & (merged_df['cmap_name'] == drug), :].mean() #.iloc[1:]
    #print(f"drug value is {drug_value}")
    if drug_value.isna().any().any():
        return []
    normalized_value = np.round(drug_value - dmso_value, 2)
    return normalized_value.tolist()

#this will choose all the available cores for multitasking
#num_cores = multiprocessing.cpu_count()
#set number of cores required for the multiprocessing
num_cores = 30

# Generating drug responses for all the drugs
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
        #print(f"length of temp+ is {temp}")
    temp = np.array(temp)
    #print(f"iter value is {iter}")
    temp = temp / iter
    print(f"shape of temp is {temp.shape}")
    print(f"size of the temp is {temp.size}")
    #print(f"temp value after converting to array is {temp}")
    #temp /= len(results)
    normalized_dict[drug] = temp
#print(f"dict  is {normalized_dict}")

#Delete untrt drug from the list since it does not have any value
for key in list(normalized_dict.keys()):
    if key == "UnTrt":
        del normalized_dict[key]

# convert dictionary to dataframe
print(f"len of norm dict is {len(normalized_dict)}")
gene_names = column_names
del gene_names[0]
#print(f"gene names {gene_names[0:5]} ")
# Filter out empty arrays from the normalized_dict
filtered_dict = {k: v for k, v in normalized_dict.items() if len(v) > 0}
"""# Iterate over the values of normalized_dict and check for empty arrays
if any(len(v) == 0 for v in normalized_dict.values()):
    print("There is at least one empty array in normalized_dict")
else:
    drug_df = pd.DataFrame(normalized_dict, index=gene_names)"""
drug_df = pd.DataFrame(filtered_dict, index=gene_names)
print(f"drug df is {drug_df.head()}")
drug_df.to_csv('/disk2/user/manche/Panacea/output/model/response/drug_responses_a549_134.csv')

# Create an empty dictionary to store the results
mean_dict = {}

# Loop through each drug pair
for i, drug1 in enumerate(filtered_dict):
    for drug2 in list(filtered_dict)[i+1:]:  # Only iterate over drug pairs where drug1 is less than drug2
        # Calculate the mean vector
        mean_vector = (filtered_dict[drug1] + filtered_dict[drug2]) / 2
        # Store the result in the mean dictionary
        mean_dict[(drug1, drug2)] = mean_vector

# Print the mean dictionary
#print(mean_dict)

# Create a new dictionary with updated keys to store the combined drug responses and then write it to csv file
updated_dict = {'__'.join(key): value for key, value in mean_dict.items()}
# Print the dictionary
#print(updated_dict)
combined_df = pd.DataFrame(updated_dict, index=gene_names)
print(combined_df.head())
combined_df.to_csv('/disk2/user/manche/Panacea/output/model/response/combined_responses_a549_134.csv')
