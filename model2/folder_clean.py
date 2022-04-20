import sys, os, shutil, PIL
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


# Data path
folder_path = os.path.abspath("../../data/cbis-ddsm")
folder_path = Path(folder_path)

for filename in os.listdir(folder_path / "tmp_neg_pos/pos"):
    shutil.move(
        folder_path / "tmp_neg_pos/pos" / filename, # source
        folder_path / "neg_pos_split/pos" / filename # destination
    )
for filename in os.listdir(folder_path / "tmp_neg_pos/neg"):
    shutil.move(
        folder_path / "tmp_neg_pos/neg" / filename, # source
        folder_path / "neg_pos_split/neg" / filename # destination
    )