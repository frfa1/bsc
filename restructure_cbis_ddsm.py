import pandas as pd

## Module to load CBIS-DDSM dataset

general_data_path = "../data/cbis-ddsm-breast-cancer-image-dataset"

meta_path = general_data_path + "/csv/meta.csv"

data_case_descriptions = [
    general_data_path + i for i in [
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv"
    ]
]

print(meta_path)

print(data_case_descriptions)

