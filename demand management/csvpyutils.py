def load_csv_dataset(csv_dir=csv_dir, window_size=None, stride=None, concat=False):
    datasets = defaultdict(list)
    files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(csv_dir, file)
        df = pd.read_csv(file_path).dropna()

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            continue

        data_tensor = torch.tensor(numeric_df.values, dtype=torch.float64)
        dataset_name = os.path.splitext(file)[0]
        descriptions = numeric_df.columns.tolist()

        dataset = UnwindowedDataset(data_tensor, dataset_name, descriptions, None)
        datasets[dataset_name].append(dataset)

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
