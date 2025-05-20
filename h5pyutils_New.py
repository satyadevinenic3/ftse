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
    '/data/TimeSeriesResearch/datasets/Satya_New'
]

H5_FILE = '/data/TimeSeriesResearch/datasets/consolidated0520.h5'


def create_h5py_dataset_from_root_dirs(root_dirs=ROOT_DIRS, h5_file=H5_FILE):
    with h5py.File(h5_file, "w") as h5f:
        for root_dir in root_dirs:
            root_prefix = os.path.basename(os.path.normpath(root_dir))

            for dataset_folder in tqdm(os.listdir(root_dir), desc=f"Datasets in {root_dir}"):
                dataset_path = os.path.join(root_dir, dataset_folder)
                if not os.path.isdir(dataset_path):
                    print("skipping", dataset_path)
                    print("\n"*50)
                    continue
                    
                #For labels
                for subfolder in os.listdir(dataset_path):
                    subfolder_path = os.path.join(dataset_path, subfolder)
                    if not os.path.isdir(subfolder_path):
                        continue

                    for csv_file in os.listdir(subfolder_path):
                        if not csv_file.lower().endswith(".csv"):
                            continue
                            
                        group_path = f"{root_prefix}/{dataset_folder}/{subfolder}"
                        group = h5f.require_group(group_path)
                        #print("group",group)

                        csv_path = os.path.join(subfolder_path, csv_file)
                        df = pd.read_csv(csv_path)

                        drop_columns = ['Unnamed: 0', 'time_sec', 'Time', 'Label']
                        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

                        column_names = df.columns.tolist()
                        data_array = df.to_numpy()
                        dataset_name = f"file_{len(group)}"
                        #print(group_path, "--", dataset_name)
                        dset = group.create_dataset(dataset_name, data=data_array)
                        dset.attrs["descriptions"] = list(column_names)
                        #print(f"Created dataset at: {group_path}/{dataset_name}")
    print("Flat HDF5 file created successfully:", h5_file)


def load_h5py_dataset(h5_file=H5_FILE, window_size=None, stride=None, concat=False):
    datasets = defaultdict(list)
    def recursive_group_traversal(group, path=""):
        for key in group.keys():
            item = group[key]

            if isinstance(item, h5py.Group):
                recursive_group_traversal(item, path + "/" + key)
            elif isinstance(item, h5py.Dataset):
                label = path.split("/")[-1]
                dataset_name = path.strip("/").split("/")[-2]

                datasets[dataset_name].append(
                    UnwindowedDataset(
                        data=item,
                        dataset_name=dataset_name,
                        descriptions=item.attrs.get("descriptions", []),
                        label=label
                    )
                )

    f = h5py.File(h5_file, 'r')
    recursive_group_traversal(f)

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
