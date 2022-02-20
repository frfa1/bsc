import pandas as pd

## Module to load CBIS-DDSM dataset

general_data_path = "../data/cbis-ddsm-breast-cancer-image-dataset"

meta_path = general_data_path + "/csv/meta.csv"

data_case_descriptions = [
    general_data_path + i for i in [
        "/csv/calc_case_description_train_set.csv",
        "/csv/calc_case_description_test_set.csv",
        "/csv/mass_case_description_train_set.csv",
        "/csv/mass_case_description_test_set.csv"
    ]
]

dfs = []
for path in data_case_descriptions:
    dfs.append(pd.read_csv(path, index_col=None, header=0))

meta = pd.read_csv(meta_path) # Loads meta data to get jpg file versions
data = pd.concat(dfs) # Loads data including labels





