import pandas as pd
import numpy as np
import os

# Helper function to get the name part without the last character and extension
def get_name_without_last_char(filename):
    return filename[:-6]  # Remove the last letter and ".mrxs"

# Load the Excel files into DataFrames
# her2_df = pd.read_excel('Her2_slides_info.xlsx')
her2_df = pd.read_excel('Her2_slides_info_with_matches_corrected.xlsx')
# he_df = pd.read_excel('HE_slides_info.xlsx')

batch_path = 'slides_data_HER2'
batch_dfs = []
for i in range(1, 7):
    batch_dfs.append(pd.read_excel(f'{batch_path}_{i}.xlsx'))
batch_df = pd.concat(batch_dfs, ignore_index=True)[['file', 'patient barcode', 'MPP', 'Her2 score']]
# # Drop rows where any column contains the value
# batch_df = batch_df[~batch_df.isin(['Missing Data']).any(axis=1)]
her2_df.dropna(inplace=True)

# Create two new columns to store the matched HE slide names and paths
her2_df['patient barcode'] = ''
her2_df['MPP'] = ''
her2_df['Her2 score'] = ''
her2_df['fold idx'] = ''


# Iterate over each file in the Her2 DataFrame
for i, her2_row in her2_df.iterrows():
    # her2_name_base = get_name_without_last_char(her2_row['SlideName'])
    her2_slidename = her2_row['SlideName']

    # Find all matching HE slides with the same base name
    # he_matches = he_df[he_df['SlideName'].str.startswith(her2_name_base)]
    batch_matches = batch_df[batch_df['file'].str.startswith(her2_slidename.split('.')[0])]

    # if not he_matches.empty:
    if not batch_matches.empty:
        # # Find the most lexicographically similar last letter
        # last_letter_her2 = her2_row['SlideName'][-6]  # Last letter before .mrxs
        # closest_he_row = \
        # min(he_matches.iterrows(), key=lambda x: abs(ord(x[1]['SlideName'][-6]) - ord(last_letter_her2)))[1]

        # # Store the matched HE slide name and path in the Her2 DataFrame
        # her2_df.at[i, 'Matched_HE_SlideName'] = closest_he_row['SlideName']
        # her2_df.at[i, 'Matched_HE_Path'] = closest_he_row['Path']
        her2_df.at[i, 'patient barcode'] = batch_matches['patient barcode'].values[0]
        her2_df.at[i, 'MPP'] = batch_matches['MPP'].values[0]
        her2_df.at[i, 'Her2 score'] = batch_matches['Her2 score'].values[0]


# 1. Get unique patient barcodes
unique_patients = her2_df['patient barcode'].unique()

# 2. Shuffle the patient barcodes
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_patients)

# 3. Assign folds
num_patients = len(unique_patients)
train_cutoff = int(num_patients * 0.75)

# First 75% for training (folds 1-5), last 25% for testing (fold 6)
folds = {patient: (i % 5 + 1 if i < train_cutoff else 6) for i, patient in enumerate(unique_patients)}

# 4. Map the fold indices back to the DataFrame
her2_df['fold idx'] = her2_df['patient barcode'].map(folds)


# Save the updated Her2 DataFrame to the original Excel file
# output_file = 'Her2_slides_info_with_matches_corrected.xlsx'
output_file = 'Her2_slides_matched_HE_folds.csv'
# her2_df.to_excel(output_file, index=False)
her2_df.to_csv(output_file, index=False)

print(f"Updated Excel file '{output_file}' has been created with matched HE slide information.")
