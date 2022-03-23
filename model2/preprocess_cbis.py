from lib2to3.pgen2.pgen import DFAState
import sys
import os
from typing import Counter
import pandas as pd
from PIL import Image

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
sys.path.insert(1, os.path.abspath("../../end2end-all-conv"))

# Github library
from dm_image import read_resize_img
import pathlib

import os.path
from os import path


def preprocess_cbis(meta):

    count = 0
    for row in meta.itertuples():

        old_img = "../" + row.new_img_location
        full_file = "../../data/cbis-ddsm/neg_pos_split/"
        filename = old_img.split("/")[-1]
        
        if row.pathology == "MALIGNANT":
            full_file += "pos/"
        else:
            full_file += "neg/"

        full_file += filename

        img_array = read_resize_img(
            old_img,
            target_size=(1152, 896),
            rescale_factor=0.003891
        )

        img = Image.fromarray(img_array)
        img = img.convert("L")
        img.save(full_file)

        count += 1
        if count % 10 == 0:
            print("Done with image #", count)

def main():
    meta = pd.read_csv("../meta_data/cbis-ddsm/test_meta_with_png.csv")
    preprocess_cbis(meta)

if __name__ == "__main__":
    main()