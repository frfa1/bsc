from lib2to3.pgen2.pgen import DFAState
import sys, os, glob, shutil
from typing import Counter
import pandas as pd
from PIL import Image

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
#sys.path.insert(1, os.path.abspath("../../end2end-all-conv"))
# Inserting parent folder in sys, to allow imports
sys.path.append("..")

# Github library
#from dm_image import read_resize_img
from load_preprocess.load_meta import get_cbis_test
import pathlib
import png

import os.path
from os import path


def preprocess_cbis(meta):

    """ Puts CBIS test images into folder with negative/positive subfolders """

    old_base = "../../data/cbis-ddsm/all_test_img"

    count = 0
    for row in meta.itertuples():

        old_img = old_base + "/" + row._1 + ".png"
        full_file = "../../data/cbis-ddsm/neg_pos_split/"
        filename = old_img.split("/")[-1]
        
        if row.pathology == "MALIGNANT":
            full_file += "pos/"
        else:
            full_file += "neg/"

        full_file += filename

        shutil.copy(
            old_img, #src_path
            full_file #dst_path
        )        

        """img_array = read_resize_img(
            old_img,
            target_size=(1152, 896),
            rescale_factor=0.003891
        )"""

        """with open(full_file, 'wb') as f:
            writer = png.Writer(height=img_array.shape[0], width=img_array.shape[1], bitdepth=8, greyscale=True)
            print(img_array.shape[0], img_array.shape[1])
            #print(writer)
            #print(img_array.tolist())
            writer.write(f, img_array.tolist())"""

        #img = Image.fromarray(img_array)
        #img = img.convert("L")
        #img.save(full_file)

        count += 1
        if count % 10 == 0:
            print("Done with image #", count)

def main():
    meta = get_cbis_test() # meta is the meta data of CBIS test
    preprocess_cbis(meta)

if __name__ == "__main__":
    main()