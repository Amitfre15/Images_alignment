import argparse
import re
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Helper function to get the name part without the last character and extension
def get_name_without_last_char(filename):
    return filename[:-6]  # Remove the last letter and ".mrxs"


def add_col_to_patient_df(file_path: str, my_file_path: str, save_file: str, add_cols: list[str], from_cols: list[str]):
    block_id_pattern = r'\d+-\d+_\d+_\d+'
    patient_df = pd.read_excel(file_path)
    her2_df = pd.read_csv(my_file_path)
    for add_col in add_cols:
        patient_df[add_col] = ''

    # for i, gil_row in gil_df.iterrows():
    for i, her2_row in her2_df.iterrows():
        her2_block_id = re.match(block_id_pattern, her2_row['SlideName'])[0]
        patient_matches = patient_df[patient_df['BlockID'].str.replace('/', '_') == her2_block_id]

        # if not her2_matches.empty:
        if not patient_matches.empty:
            for add_col, from_col in zip(add_cols, from_cols):
                patient_df.at[patient_matches.index[0], add_col] = her2_row[from_col]
        else:
            print(her2_block_id)

    patient_df.to_excel(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def mark_matching_blocks(file_path: str, my_file_path: str, save_file: str):
    add_col = 'has_pair'
    block_id_pattern = r'\d+-\d+_\d+_\d+'
    gil_df = pd.read_excel(file_path)
    her2_df = pd.read_csv(my_file_path)
    her2_df['MatchedPattern'] = her2_df['SlideName'].str.extract(f'({block_id_pattern})', expand=False)
    gil_df[add_col] = ''
    counter = 0
    pattern_counts = her2_df['MatchedPattern'].value_counts()

    # for i, gil_row in gil_df.iterrows():
    for i, her2_row in her2_df.iterrows():
        her2_block_id = re.match(block_id_pattern, her2_row['SlideName'])[0]
        gil_matches = gil_df[gil_df['BlockID'].str.replace('/', '_') == her2_block_id]
        # block_id = gil_row['BlockID'].replace('/', '_')
        # her2_matches = her2_df[her2_df['SlideName'].str.startswith(block_id)]

        # if not her2_matches.empty:
        if not gil_matches.empty:
            if gil_df.at[i, add_col] == 1:
                print(f'{her2_block_id} again')
            gil_df.at[i, add_col] = 1
            counter += 1
        else:
            print(her2_block_id)

    print(counter)
    # gil_df.to_excel(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def update_excel_file(file_path: str, save_file: str, output_dirs):
    # Load the Excel files into DataFrames
    # her2_df = pd.read_excel('Her2_slides_info.xlsx')
    # her2_df = pd.read_excel('Her2_slides_info_with_matches_corrected.xlsx')
    if file_path.endswith('xlsx'):
        her2_df = pd.read_excel(file_path)
    else:
        her2_df = pd.read_csv(file_path)

    # he_df = pd.read_excel('HE_slides_info.xlsx')

    # dirs = ['IHC_to_Her2_score', 'IHC_to_Her2_status', 'HE_to_Her2_score', 'HE_to_Her2_status']
    # batch_path = 'slides_data_HER2'
    batch_dfs = {}
    for output_dir in output_dirs:
        her2_df[output_dir] = ''
        batch_dfs[output_dir] = []
        full_output_dir = os.path.join(os.getcwd(), 'outputs', output_dir)
        config = 'her2' if output_dir.endswith('score') else 'her2_status'
        op_dir_w_cfg = os.path.join(full_output_dir, config)
        infer_dirs = [d for d in os.listdir(op_dir_w_cfg) if 'infer' in d and not d.endswith('.out')]
        print(f'infer_dirs = {infer_dirs}')
        # for i in range(1, 7):
        for idir in infer_dirs:
            csv_path = os.path.join(full_output_dir, config, idir, f'eval_pretrained_{config}', 'inference_results',
                                    'slide_scores.csv')
            # batch_dfs.append(pd.read_excel(f'{batch_path}_{i}.xlsx'))
            batch_dfs[output_dir].append(pd.read_csv(csv_path))
            print(f'batch_dfs[output_dir] = {batch_dfs[output_dir]}')
        batch_dfs[output_dir] = pd.concat(batch_dfs[output_dir], ignore_index=True)[['slide_name', 'score']]
    # batch_df = pd.concat(batch_dfs, ignore_index=True)[['file', 'patient barcode', 'MPP', 'Her2 score']]
    # # Drop rows where any column contains the value
    # batch_df = batch_df[~batch_df.isin(['Missing Data']).any(axis=1)]
    her2_df.dropna(inplace=True)

    # Create new columns to store the matched HE slide names and paths
    # her2_df['patient barcode'] = ''
    # her2_df['MPP'] = ''
    # her2_df['Her2 score'] = ''
    # her2_df['fold idx'] = ''

    # Iterate over each file in the Her2 DataFrame
    for i, her2_row in her2_df.iterrows():
        # her2_name_base = get_name_without_last_char(her2_row['SlideName'])

        # Find all matching HE slides with the same base name
        # he_matches = he_df[he_df['SlideName'].str.startswith(her2_name_base)]
        for key, batch_df in batch_dfs.items():
            slide_key = "SlideName" if key.startswith('IHC') else "Matched_HE_SlideName"
            her2_slidename = her2_row[slide_key]
            batch_matches = batch_df[batch_df['slide_name'].str.startswith(her2_slidename.split('.')[0])]

            # if not he_matches.empty:
            if not batch_matches.empty:
                # # Find the most lexicographically similar last letter
                # last_letter_her2 = her2_row['SlideName'][-6]  # Last letter before .mrxs
                # closest_he_row = \
                # min(he_matches.iterrows(), key=lambda x: abs(ord(x[1]['SlideName'][-6]) - ord(last_letter_her2)))[1]

                # # Store the matched HE slide name and path in the Her2 DataFrame
                # her2_df.at[i, 'Matched_HE_SlideName'] = closest_he_row['SlideName']
                # her2_df.at[i, 'Matched_HE_Path'] = closest_he_row['Path']
                # her2_df.at[i, 'patient barcode'] = batch_matches['patient barcode'].values[0]
                # her2_df.at[i, 'MPP'] = batch_matches['MPP'].values[0]
                # her2_df.at[i, 'Her2 score'] = batch_matches['Her2 score'].values[0]
                her2_df.at[i, key] = batch_matches['score'].values[0]

    # Split the df to folds
    # # 1. Get unique patient barcodes
    # unique_patients = her2_df['patient barcode'].unique()
    #
    # # 2. Shuffle the patient barcodes
    # np.random.seed(42)  # For reproducibility
    # np.random.shuffle(unique_patients)
    #
    # # 3. Assign folds
    # num_patients = len(unique_patients)
    # train_cutoff = int(num_patients * 0.75)
    #
    # # First 75% for training (folds 1-5), last 25% for testing (fold 6)
    # folds = {patient: (i % 5 + 1 if i < train_cutoff else 6) for i, patient in enumerate(unique_patients)}
    #
    # # 4. Map the fold indices back to the DataFrame
    # her2_df['fold idx'] = her2_df['patient barcode'].map(folds)

    # Save the updated Her2 DataFrame to the output file
    # output_file = 'Her2_slides_info_with_matches_corrected.xlsx'
    # output_file = 'Her2_slides_matched_HE_folds_infer.csv'
    # her2_df.to_excel(output_file, index=False)
    her2_df.to_csv(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def compute_metrics(file_paths, label_cols, score_cols, plot_labels, save_dir=None):
    y_trues1, y_trues2, y_scores = [], [], []
    for file_path, label_col, score_col in zip(file_paths, label_cols, score_cols):
        her2_df = pd.read_csv(file_path)
        valid_indices = her2_df.index[
            (her2_df[label_col] != "Missing Data") & (~her2_df[score_col].isna())].tolist()
        y_true = her2_df[label_col][valid_indices].values.astype(float)
        # multiclass
        if len(np.unique(y_true)) > 2:
            # Case 1: [0, 0.5, 1] vs [2, 3]
            binary_labels_case1 = np.isin(y_true, [2, 3]).astype(int)
            y_trues1.append(binary_labels_case1)
            # Case 2: [0, 0.5, 1, 2] vs [3]
            binary_labels_case2 = np.isin(y_true, [3]).astype(int)
            y_trues2.append(binary_labels_case2)
        else:
            y_trues1.append(y_true)
        y_score = her2_df[score_col][valid_indices].values
        y_scores.append(y_score)

    if len(y_trues2) > 0:
        show_roc_and_calc_auc(y_trues=y_trues1, y_scores=y_scores, score_title=f'[0, 0.5, 1] vs [2, 3] {score_cols[0]}',
                              plot_labels=plot_labels, save_dir=save_dir)
        show_roc_and_calc_auc(y_trues=y_trues2, y_scores=y_scores, score_title=f'[0, 0.5, 1, 2] vs [3] {score_cols[0]}',
                              plot_labels=plot_labels, save_dir=save_dir)
    else:
        show_roc_and_calc_auc(y_trues=y_trues1, y_scores=y_scores, score_title=score_cols[0], plot_labels=plot_labels,
                              save_dir=save_dir)


def show_roc_and_calc_auc(y_trues, y_scores, score_title, plot_labels, save_dir=None):
    plt.figure(figsize=(8, 6))
    for y_true, y_score, plot_label in zip(y_trues, y_scores, plot_labels):
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Calculate AUROC
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.plot(fpr, tpr, label=f'{plot_label} ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{score_title} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{score_title} ROC Curve.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Search and replace file names and content based on Excel mapping.")
    # parser.add_argument('-r', '--root', required=True, help='Root directory to search for files.')
    parser.add_argument('-u', '--update_file', action='store_true', help='Whether to update the file')
    parser.add_argument('-f', '--file_paths', required=True, nargs='+', help='Excel file to update')
    parser.add_argument('-sf', '--second_file_path', type=str, help='Second excel file to update')
    parser.add_argument('-sd', '--save_dir', type=str, help='Directory to save metrics plots')
    parser.add_argument('-c', '--compute_metrics', action='store_true', help='Whether to compute metrics')
    parser.add_argument('-pl', '--plot_labels', nargs='+', help='Curve labels to show in the metrics plot')
    parser.add_argument('-l', '--label_column', nargs='+', help='Name of the label column in the file')
    parser.add_argument('-s', '--score_column', nargs='+', help='Name of the predicted score column')
    # parser.add_argument('-a', '--add_columns', nargs='+', help='Names of columns to add')
    parser.add_argument('-fr', '--from_columns', nargs='+', help='Names of columns to take values from')
    parser.add_argument('-m', '--mark_matches', action='store_true', help='Whether to mark existing blocks')
    parser.add_argument('-p', '--patient_df_update', action='store_true', help='Whether to update patient df')

    # example command: -f ./excel_files/Her2_slides_matched_HE_folds_infer.csv -sf ./excel_files/carmel_per_block_marked.xlsx -fr IHC_to_Her2_score IHC_to_Her2_status HE_to_Her2_score HE_to_Her2_status -p
    args = parser.parse_args()
    print(f'args = {args}')

    # her2_csv_path = os.path.join(os.getcwd(), 'workspace', 'WSI', 'metadata_csvs', 'Her2_slides_matched_HE_folds.csv')
    her2_csv_path = args.file_paths
    if args.update_file:
        update_excel_file(file_path=her2_csv_path)

    # her2_csv_path = os.path.join('excel_files', 'Her2_slides_matched_HE_folds_infer.csv')
    if args.compute_metrics:
        if args.label_column is not None and args.score_column is not None:
            compute_metrics(file_paths=her2_csv_path, label_cols=args.label_column, score_cols=args.score_column,
                            plot_labels=args.plot_labels, save_dir=args.save_dir)
        else:
            print(f'Please specify label_column and score_column for metrics calculation.\n'
                  f'label_column = {args.label_column}, score_column = {args.score_column}')

    if args.mark_matches:
        if args.second_file_path is not None:
            mark_matching_blocks(file_path=args.second_file_path, my_file_path=her2_csv_path,
                                 save_file=f'{args.second_file_path.split(".xlsx")[0]}_marked.xlsx')

    if args.patient_df_update:
        if args.second_file_path is not None:
            add_cols = [f'mpp1_{fr_col}' for fr_col in args.from_columns]
            add_col_to_patient_df(file_path=args.second_file_path, my_file_path=her2_csv_path,
                                  save_file=f'{args.second_file_path.split("_marked.xlsx")[0]}.xlsx',
                                  add_cols=add_cols, from_cols=args.from_columns)


if __name__ == '__main__':
    main()
