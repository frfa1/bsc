from lib2to3.pgen2.pgen import DFAState
import sys
import os
import pandas as pd

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
sys.path.insert(1, os.path.abspath("../breast_cancer_classifier"))

# Github library
import src.cropping.crop_single as crop_single
import pathlib

import os.path
from os import path

def crop_column(new_img_location, subject_id):

    global count
    count += 1
    if count % 10 == 0:
        print(count)

    subject_id2 = subject_id.split("_")

    """crop_single.crop_single_mammogram(
        mammogram_path = new_img_location,
        horizontal_flip = "NO",
        view = subject_id2[-2][0] + "-" + subject_id2[-1],
        cropped_mammogram_path = cropped_img_path + "/" + subject_id + ".png",
        metadata_path = metadata_path_after_crop + "/" + subject_id + ".pkl",
        num_iterations = 100,
        buffer_size = 50
    )"""
    #print([cropped_img_path + "/" + subject_id + ".png", metadata_path_after_crop + "/" + subject_id + ".pkl"])
    return cropped_img_path + "/" + subject_id + ".png"


def main():
    meta = pd.read_csv(
        "meta_data/cbis-ddsm/test_meta_with_png.csv"
    )
    
    global cropped_img_path; global metadata_path_after_crop; global count
    cropped_img_path = "../data/cbis-ddsm/all_test_img_output/cropped"
    metadata_path_after_crop = "../data/cbis-ddsm/all_test_img_output/metadata"
    count = 0

    """meta[["cropped_png_location", "metadata_path_after_crop"]] = meta.apply(
        lambda row: crop_column(
            row["new_img_location"],
            #row["image view"],
            row["Subject ID"]
        ), axis=1
    )"""

    meta["cropped_png_location"] = meta.apply(
        lambda row: crop_column(
            row["new_img_location"],
            #row["image view"],
            row["Subject ID"]
        ), axis=1
    )


    # Save meta data with cropped image locations and metadata locations
    meta.to_csv(
        "meta_data/cbis-ddsm/test_meta_after_crop.csv",
        index = False,
    )

if __name__ == "__main__":
    main()

