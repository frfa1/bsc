import sys, os, shutil, PIL
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

# Data path
folder_path = os.path.abspath("../../data/cbis-ddsm")
folder_path = Path(folder_path)

# Meta
meta = pd.read_csv(
    "../meta_data/cbis-ddsm/test_meta_with_features.csv"
)

# Arg 1 as BIRAD
if len(sys.argv) > 1:
    birad = int(sys.argv[1])
else:
    birad = None

if birad or birad==0:
    meta = meta[meta["assessment"] == birad]

# Arg 2 as feature
if len(sys.argv) > 2:
    feature = sys.argv[2]
else:
    feature = None

if feature:
    meta = meta.loc[meta[feature] != 1]

if not os.path.exists(folder_path / 'tmp_neg_pos'):
    os.makedirs(folder_path / 'tmp_neg_pos')
if not os.path.exists(folder_path / 'tmp_neg_pos/pos'):
    os.makedirs(folder_path / 'tmp_neg_pos/pos')
if not os.path.exists(folder_path / 'tmp_neg_pos/neg'):
    os.makedirs(folder_path / 'tmp_neg_pos/neg')

img_list = meta[["Subject ID"]].values.flatten().tolist()
print(img_list)

for img in img_list:
    img_name = img + ".png"
    if os.path.exists(folder_path / "neg_pos_split/pos" / img_name):
        shutil.move(
            folder_path / "neg_pos_split/pos" / img_name, # source
            folder_path / "tmp_neg_pos/pos" / img_name # destination
        )
    elif os.path.exists(folder_path / "neg_pos_split/neg" / img_name):
        shutil.move(
            folder_path / "neg_pos_split/neg" / img_name, # source
            folder_path / "tmp_neg_pos/neg" / img_name # destination
        )

# Save meta in temp directory
meta.to_csv(
    folder_path / 'tmp_neg_pos' / 'tmp_meta.csv',
    index = False,
)