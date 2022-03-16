import pickle
import json
import pandas as pd
import os

img_basepath = "../data/cbis-ddsm/all_test_img"

data = []




directory = os.fsencode(img_basepath)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"): 
        print(filename)
        continue
    else:
        continue

# Loadingm meta data
meta = pd.read_csv(
    "meta_data/cbis-ddsm/test_meta_with_png.csv"
)

# Get unique patients
unique_patients = meta.loc[:, "patient_id"].unique()

for patient in unique_patients:
    meta






#print(meta.loc[:,"Subject ID"])
#print(len(meta.loc[:, "patient_id"].unique()))

#output = open('exam_list_before_cropping.pkl', 'wb')
#pickle.dump(data, output)
#output.close()

#[
#    {'horizontal_flip': 'NO', 'L-CC': ['0_L_CC'], 'L-MLO': ['0_L_MLO'], 'R-MLO': ['0_R_MLO'], 'R-CC': ['0_R_CC']}
#]