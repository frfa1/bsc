import pandas as pd
import numpy as np
import os

def get_cbis_test(whole_image_labels=True):
    basepath = "../../data/cbis-ddsm"

    # Get specific test meta data
    calc_test = pd.read_csv(basepath + "/calc_case_description_test_set.csv")
    mass_test = pd.read_csv(basepath + "/mass_case_description_test_set.csv")

    # Align their columns
    calc_test = calc_test.rename(columns={
        "breast density":"breast_density",
        })

    # Concat test cases
    all_test = pd.concat([calc_test, mass_test])
    all_test["Subject ID"] = all_test["image file path"].str.split("/").str[0]

    # Join with general meta data
    meta = pd.read_csv(basepath + "/manifest-ZkhPvrLo5216730872708713142/metadata2.csv")
    all_meta = all_test.merge(meta, on="Subject ID")

    if whole_image_labels:
        # Labeling whole images: Malignant if any malignant ROI, benign otherwise
        # Returns just Subject ID and pathology for whole image
        all_meta["true_benign"], meta["true_malignant"] = 0, 0
        pos_cases = all_meta[all_meta["pathology"] == "MALIGNANT"]["Subject ID"] # Get an iterable with all cases that contains a positive (malignant) segment
        all_meta["true_malignant"] = np.where(all_meta["Subject ID"].isin(pos_cases), 1, 0)
        all_meta["true_benign"] = np.where(all_meta["Subject ID"].isin(pos_cases), 0, 1)
        all_meta.drop_duplicates(subset=["Subject ID"], inplace=True) # Removes duplicates
        all_meta = all_meta[["Subject ID", "pathology", "true_malignant", "true_benign"]]

    return all_meta

def get_cbis_train(whole_image_labels=True):
    basepath = "../../data/cbis-ddsm"

    # Get specific test meta data
    calc_test = pd.read_csv(basepath + "/calc_case_description_train_set.csv")
    mass_test = pd.read_csv(basepath + "/mass_case_description_train_set.csv")

    # Align their columns
    calc_test = calc_test.rename(columns={
        "breast density":"breast_density",
        })

    # Concat test cases
    all_test = pd.concat([calc_test, mass_test])
    all_test["Subject ID"] = all_test["image file path"].str.split("/").str[0]

    # Join with general meta data
    meta = pd.read_csv(basepath + "/manifest-ZkhPvrLo5216730872708713142/metadata2.csv")
    all_meta = all_test.merge(meta, on="Subject ID")

    if whole_image_labels:
        # Labeling whole images: Malignant if any malignant ROI, benign otherwise
        # Returns just Subject ID and pathology for whole image
        all_meta["true_benign"], meta["true_malignant"] = 0, 0
        pos_cases = all_meta[all_meta["pathology"] == "MALIGNANT"]["Subject ID"] # Get an iterable with all cases that contains a positive (malignant) segment
        all_meta["true_malignant"] = np.where(all_meta["Subject ID"].isin(pos_cases), 1, 0)
        all_meta["true_benign"] = np.where(all_meta["Subject ID"].isin(pos_cases), 0, 1)
        all_meta.drop_duplicates(subset=["Subject ID"], inplace=True) # Removes duplicates
        all_meta = all_meta[["Subject ID", "pathology", "true_malignant", "true_benign"]]

    return all_meta

def get_inbreast_medical_report(meta):
    basepath = "../../data/inbreast"
    img_folder = basepath + "/png_versions"
    medical_reports_folder = basepath + "/MedicalReports"

    meta["Full File Name"] = 0
    meta["Patient"] = 0
    meta["Medical Report"] = ""

    for filename in os.listdir(img_folder):
        if not filename.endswith(".png"): # Skips non-images
            continue

        name = filename.split("_")[0]
        patient = filename.split("_")[1]

        for idx, row in meta.iterrows():
            if name == str(row["File Name"]):
                #print(row.Index)
                meta.loc[idx, "Full File Name"] = filename
                meta.loc[idx, "Patient"] = patient

    for medfile in os.listdir(medical_reports_folder):
        if not medfile.endswith(".txt"): # Skips non-txt
            continue

        full_file = os.path.join(medical_reports_folder, medfile)

        patient = medfile.split(".")[0].split("_")[0]

        #print(medfile)
        #print(patient)

        with open(full_file, 'r', encoding='latin-1') as f:
            lines = f.read()

        meta.loc[meta["Patient"] == patient, "Medical Report"] = meta.loc[meta["Patient"] == patient, "Medical Report"] + "\n" + lines

    # Save to csv
    meta.to_csv( 
        basepath + "/INbreast_with_medical.csv",
        index=False
    )


def get_inbreast(full=True, whole_image_labels=True):
    basepath = "../../data/inbreast"

    if not full:
        # Get metadata
        meta = pd.read_csv(basepath + "/INbreast_modified.csv")
    else:
        meta = pd.read_csv(basepath + "/INbreast_with_medical.csv")

    #print(meta["Bi-Rads"].unique())
    
    if whole_image_labels:
        meta["true_benign"], meta["true_malignant"] = 0, 0
        pos_cases = ["4", "4a", "4b", "4c", "5", "6"] # 4, 5 and 6 as positive
        meta["true_malignant"] = np.where(meta["Bi-Rads"].isin(pos_cases), 1, 0)
        meta["true_benign"] = np.where(meta["Bi-Rads"].isin(pos_cases), 0, 1)

    return meta

#def get_translated_inbreast():


"""def get_inbreast_test():
    meta = get_inbreast(full=False, whole_image_labels=False)
    get_inbreast_medical_report(meta)
    meta = get_inbreast()"""


def main():
    """meta = get_inbreast(full=False, whole_image_labels=False)
    get_inbreast_medical_report(meta)
    meta = get_inbreast()"""

    meta = get_cbis_test(whole_image_labels = False)
    print(meta)
    

if __name__ == "__main__":
    main()