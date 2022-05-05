from lib2to3.pgen2.pgen import DFAState
import sys, os, glob, shutil
from typing import Counter
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import copy
import numpy as np

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
#sys.path.insert(1, os.path.abspath("../../end2end-all-conv"))
# Inserting parent folder in sys, to allow imports
sys.path.append("..")

# Github library
#from dm_image import read_resize_img
from load_preprocess.load_meta import get_inbreast
import pathlib
import png

import os.path
from os import path
from pathlib import Path

def generate_split():
    """ Generates train/test split and saves file """

    basepath = "../../data/INbreast"

    meta = get_inbreast()
    #(meta.columns)
    #print(meta)
    meta = meta[meta["Bi-Rads"] != "3"] # Remove birads 3
    meta.loc[(meta["Bi-Rads"] == "4a")|(meta["Bi-Rads"] == "4b")|(meta["Bi-Rads"] == "4c"), "Bi-Rads"] = "4" # Streamline birads 4

    """train, test = train_test_split(
        meta, test_size=0.2, random_state=42, stratify=meta["Bi-Rads"]
    )""" # Train-test split - but doesn't take patients into account

    birads = meta["Bi-Rads"].unique()
    """patients = meta["Patient"].unique()
    random.shuffle(patients) # Shuffle patient list before initial split
    print(patients)
    n_patients = len(patients)
    n_train, n_test = round(n_patients*0.8), round(n_patients*0.2)"""

    # create empty train, val and test datasets
    dict_train = {
        "File Name": [],
        "Bi-Rads": [],
        "Patient": [],
        "Full File Name": []
    }
    dict_val= {
        "File Name": [],
        "Bi-Rads": [],
        "Patient": [],
        "Full File Name": []
    }
    dict_test = {
        "File Name": [],
        "Bi-Rads": [],
        "Patient": [],
        "Full File Name": []
    }

    birad_dict = {}
    for birad in birads:
        birad_meta = meta.loc[(meta["Bi-Rads"] == birad)]
        birad_meta = birad_meta.sample(frac=1)
        n_birad = len(birad_meta)
        n_train, n_val, n_test = round(n_birad*0.6), round(n_birad*0.2), round(n_birad*0.2)
        birad_dict[birad] = {
            "n_train": n_train,
            "n_val": n_val, 
            "n_test": n_test, 
        }  

    meta = meta.sample(frac=1) # Shuffle meta
    #meta['count'] = meta.groupby('Bi-Rads')['Bi-Rads'].transform(pd.Series.value_counts)
    meta['count'] = meta['Bi-Rads'].map(meta['Bi-Rads'].value_counts())
    meta.sort_values('count', ascending=True, inplace=True) # Sort by birads, to handle small sample birads first

    for idx, row in meta.iterrows():

        # Skips rows that have already been added to either train, val or test
        if (row["File Name"] in dict_train["File Name"]) or (row["File Name"] in dict_val["File Name"]) or (row["File Name"] in dict_test["File Name"]):
            continue

        train_birad = len([y for y in dict_train["Bi-Rads"] if y == row["Bi-Rads"]]) # Gets the number of cases with the birad of the current row per train/val/test.
        val_birad = len([y for y in dict_val["Bi-Rads"] if y == row["Bi-Rads"]])
        test_birad = len([y for y in dict_test["Bi-Rads"] if y == row["Bi-Rads"]])
        m_train = birad_dict[row["Bi-Rads"]]["n_train"] - train_birad # The total "goal" birad cases in the set minus the current birad cases in the set
        m_val = birad_dict[row["Bi-Rads"]]["n_val"] - val_birad
        m_test = birad_dict[row["Bi-Rads"]]["n_test"] - test_birad

        if m_train > 0 and not (val_birad == 0 or test_birad == 0): # Checks if the set needs more of the current birad. Only fill, if val and test are not 0 for the birad
            patient_df = meta[meta["Patient"] == row["Patient"]] # Index the dataframe with all the cases for that patient
            for idx2, row2 in patient_df.iterrows(): # Add all images from that patient to the same group
                if row2["File Name"] not in dict_train["File Name"]: # ... Only if image not already in group
                    dict_train["Patient"].append(row2["Patient"])
                    dict_train["Bi-Rads"].append(row2["Bi-Rads"])
                    dict_train["File Name"].append(row2["File Name"])
                    dict_train["Full File Name"].append(row2["Full File Name"])

        elif m_val > 0 and not (test_birad == 0):
            patient_df = meta[meta["Patient"] == row["Patient"]]
            for idx2, row2 in patient_df.iterrows():
                if row2["File Name"] not in dict_val["File Name"]:
                    dict_val["Patient"].append(row2["Patient"])
                    dict_val["Bi-Rads"].append(row2["Bi-Rads"])
                    dict_val["File Name"].append(row2["File Name"])
                    dict_val["Full File Name"].append(row2["Full File Name"])

        elif m_test > 0:
            patient_df = meta[meta["Patient"] == row["Patient"]]
            for idx2, row2 in patient_df.iterrows():
                if row2["File Name"] not in dict_test["File Name"]:
                    dict_test["Patient"].append(row2["Patient"])
                    dict_test["Bi-Rads"].append(row2["Bi-Rads"])
                    dict_test["File Name"].append(row2["File Name"])
                    dict_test["Full File Name"].append(row2["Full File Name"])

        else: # In case of leftovers, add them to train
            if row["File Name"] not in dict_train["File Name"]:
                dict_train["Patient"].append(row["Patient"])
                dict_train["Bi-Rads"].append(row["Bi-Rads"])
                dict_train["File Name"].append(row["File Name"])
                dict_train["Full File Name"].append(row["Full File Name"])

    df_train = pd.DataFrame(dict_train)
    df_val = pd.DataFrame(dict_val)
    df_test = pd.DataFrame(dict_test)
    
    df_train["Split"] = "train"
    df_val["Split"] = "val"
    df_test["Split"] = "test"

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print("SANITY CHECK:")
    print("Length original & with split:", len(meta), len(df_all))
    print("And unique filenames:", meta["File Name"].nunique(), df_all["File Name"].nunique())

    merged_df = df_all.merge(meta, on=["File Name","Bi-Rads","Patient","Full File Name"])
    print("Merged with meta data:")
    print(merged_df)


    print("\n")

    print("-- TRAIN --")
    print("Percentage:", len(df_train)/len(meta)*100)
    print(df_train["Bi-Rads"].value_counts())
    print(df_train["Bi-Rads"].value_counts(normalize=True))
    print("\n")

    print("-- VAL --")
    print("Percentage:", len(df_val)/len(meta)*100)
    print(df_val["Bi-Rads"].value_counts())
    print(df_val["Bi-Rads"].value_counts(normalize=True))
    print("\n")

    print("-- TEST --")
    print("Percentage:", len(df_test)/len(meta)*100)
    print(df_test["Bi-Rads"].value_counts())
    print(df_test["Bi-Rads"].value_counts(normalize=True))
    print("\n")

    # Save to csv
    merged_df.to_csv( 
        basepath + "/INbreast_with_split.csv",
        index=False
    )

def preprocess_inbreast():
    """ Puts INbreast data into folders with negative/positive subfolders """

    basepath = "../../data/INbreast"
    old_base = basepath + "/png_versions"
    meta = pd.read_csv(basepath + "/INbreast_with_split.csv")

    # Prepare empty directories
    path = os.path.join(basepath, "train/pos")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    path = os.path.join(basepath, "train/neg")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    path = os.path.join(basepath, "val/pos")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    path = os.path.join(basepath, "val/neg")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    path = os.path.join(basepath, "test/pos")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    path = os.path.join(basepath, "test/neg")
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    
    # Move images to directories
    count = 0
    for row in meta.itertuples():
        old_img = old_base + "/" + row._4
        full_file = basepath + "/" + row.Split + "/"
        filename = old_img.split("/")[-1]

        if row.true_malignant == 1:
            full_file += "pos/"
        else:
            full_file += "neg/"

        full_file += filename

        shutil.copy(
            old_img, #src_path
            full_file #dst_path
        )

def get_pixel_mean():
    """ Gets pixel mean of TRAIN set of INBreast """

    # Data path
    folder_path = os.path.abspath("../../data/INbreast/train")
    folder_path = Path(folder_path)

    # Get list of images
    images = []
    for neg_img in os.listdir(folder_path / "neg"):
        images.append(os.path.join(folder_path / "neg/", neg_img))
    for pos_img in os.listdir(folder_path / "pos"):
        images.append(os.path.join(folder_path / "pos/", pos_img))
    # Get the mean
    current_mean = 0
    pixel_counter = 0
    for img in images:
        imarr = np.array(Image.open(img),dtype=np.float)
        pixel_counter += len(imarr)
        current_mean = current_mean + (np.mean(imarr) * len(imarr))
    final_mean = current_mean / pixel_counter * 0.003891

    #print("Data input pixel mean", final_mean)
    return final_mean # 29.197728008670026



def main():
    #generate_split()
    #preprocess_inbreast()
    mean = get_pixel_mean()
    print(mean)

if __name__ == "__main__":
    main()