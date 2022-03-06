
import pandas as pd
import sys, os

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
sys.path.insert(1, os.path.abspath("../breast_cancer_classifier"))

# Github library
import src.optimal_centers.get_optimal_center_single as center_single

def extract_centers_column(cropped_png_location, subject_id):

    global count
    count += 1
    if count % 10 == 0:
        print(count)

    # Metadata file of the observation after cropping
    metadata_path = metadata_path_after_crop + "/" + subject_id + ".pkl"

    # Get optimal center of the observation
    center_single.get_optimal_center_single(
        cropped_mammogram_path = cropped_png_location,
        metadata_path = metadata_path
    )

def main():
    meta = pd.read_csv(
        "meta_data/cbis-ddsm/test_meta_after_crop.csv"
    )

    global count, metadata_path_after_crop
    count = 0
    metadata_path_after_crop = "../data/cbis-ddsm/all_test_img_output/metadata"

    meta.apply(
        lambda row: extract_centers_column(
            row["cropped_png_location"],
            row["Subject ID"]
        ), axis=1
    )

if __name__ == "__main__":
    main()