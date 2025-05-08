from collections import defaultdict
from ftse.data.Dataset import UnwindowedDataset
from tqdm.auto import tqdm
import os
import h5py
import numpy as np
import pandas as pd
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


ROOT_DIRS = [
    '/data/TimeSeriesResearch/datasets/c3server_labelled_separated/',
    '/data/TimeSeriesResearch/datasets/liu_ice_processed_labelled_separated/',
    '/data/TimeSeriesResearch/datasets/kaggle/processed/processed_datasets_labelled_separated/',
    '/data/TimeSeriesResearch/datasets/Satya/'
]

H5_FILE = '/data/TimeSeriesResearch/datasets/satya_data1.h5'


def create_h5py_dataset_from_root_dirs(root_dirs=ROOT_DIRS, h5_file=H5_FILE):
    with h5py.File(h5_file, "w") as h5f:
        # Iterate over each root directory.
        for root_dir in root_dirs:
            print("root", root_dir)
            # Use the basename of the root directory as a prefix
            root_prefix = os.path.basename(os.path.normpath(root_dir))
            # Iterate over dataset folders in the current root directory.
            for dataset_folder in tqdm(os.listdir(root_dir), desc=f"Datasets in {root_dir}"):
                print("dataset_folder", dataset_folder)
                dataset_path = os.path.join(root_dir, dataset_folder)
                if not os.path.isdir(dataset_path):
                    print("skipping", dataset_path)
                    print("\n"*50)
                    continue  # Skip non-directory items

                # Dictionary to track file indices for each label within this dataset folder.
                label_counts = {}

                # Iterate over each CSV file in the current dataset folder.
                for csv_file in tqdm(os.listdir(dataset_path), desc=dataset_folder, leave=False):
                    # Process only CSV files.
                    if not csv_file.lower().endswith(".csv"):
                        continue

                    # Example filename: "file_5_Normal Operation.csv"
                    # Extract label: take everything after the second underscore and strip ".csv".
                    parts = csv_file.split('_')
                    if len(parts) >= 3:
                        label = '_'.join(parts[2:]).replace('.csv', '')
                    else:
                        label = "Unknown"

                    # Initialize or update the file index counter for this label.
                    if label not in label_counts:
                        label_counts[label] = 0
                    file_idx = label_counts[label]
                    label_counts[label] += 1

                    # Read the CSV file.
                    csv_path = os.path.join(dataset_path, csv_file)
                    df = pd.read_csv(csv_path)
                    # Drop unwanted columns.
                    cols_to_drop = ['Unnamed: 0', 'time_sec', 'Time', 'Label']
                    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
                    # Save the column names as metadata.
                    column_names = df.columns.tolist()
                    data_array = df.to_numpy()

                    # Create a flat key in the format:
                    # "root_prefix___dataset_folder___label___file_idx"
                    key = f"{dataset_folder}___{label}___{file_idx}"
                    dset = h5f.create_dataset(key, data=data_array)
                    dset.attrs["descriptions"] = list(column_names)

    print("Flat HDF5 file created successfully:", h5_file)


def load_h5py_dataset(h5_file=H5_FILE, window_size=None, stride=None, concat=False):
    datasets = defaultdict(list)
    f = h5py.File(h5_file, 'r')
    for key in f.keys():
        dataset_name, label, file_idx = key.split('___')
        datasets[dataset_name].append(
            UnwindowedDataset(
                data=f[key],
                dataset_name=dataset_name,
                descriptions=f[key].attrs["descriptions"],
                label=label
            )
        )

    if window_size and stride:
        for dataset_name in datasets:
            datasets[dataset_name] = [
                dataset.window(window_size=window_size, stride=stride)
                for dataset in datasets[dataset_name]
            ]

    if concat:
        for dataset_name in datasets:
            datasets[dataset_name] = torch.utils.data.ConcatDataset(
                datasets[dataset_name]
            )

    return datasets

if __name__ == "__main__":
    create_h5py_dataset_from_root_dirs()