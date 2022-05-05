import pandas as pd
import pathlib
import sys, os, glob, shutil
from PIL import Image

from preprocess.dicom_to_png16 import save_dicom_image_as_png

# Options to display Pandas dataframes in terminal
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 350)
#pd.set_option("display.max_colwidth", 400)

def restructure_kau_bcmd():
    # Basepath for the dataset
    basepath = "../data/kau-bcmd"
    all_img = basepath + "/all_img"

    if not os.path.exists(all_img):
        os.mkdir(all_img)
        print("Directory " , all_img ,  " Created ")
    else:    
        print("Directory " , all_img ,  " already exists")

    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(basepath):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    
    for f in listOfFiles:
        if f.endswith(".jpg"):
            print(f)
            #filename = f.split("/")[-1].split(".")[0].replace(" ", "") + ".png"
            filename = f.split("/")[-1].replace(" ", "")
            shutil.copy(
                f, #src_path
                os.path.join(all_img, filename) #dst_path
            )
            """im1 = Image.open(f)
            im1.save(
                os.path.join(all_img, filename)
            )"""

    #im1 = Image.open(filename)
    #im1.save(r'path where the PNG will be stored\new file name.png')




    return

    # Get all the meta data
    meta = pd.read_csv(basepath + "/manifest-ZkhPvrLo5216730872708713142/metadata2.csv")
    all_meta = all_test.merge(meta, on="Subject ID")

    # Remove cases with ambiguous labels, as those correspond to ROI-level labels (rather than whole image)
    counts = all_meta.groupby(["image file path"])["pathology"].nunique().reset_index(name='count') \
                                .sort_values(['count'], ascending=False)
    multi_label_list = list(counts[counts["count"] > 1]["image file path"])
    all_meta = all_meta[~all_meta["image file path"].isin(multi_label_list)]

    # Remove duplicate by Subject ID. Note: Subject ID contains patient ID, view and L/R.
    # It is the same as removing duplicates by File Name.
    all_meta.drop_duplicates(subset=["Subject ID"], inplace=True)

    # New image folder // For PNGs
    new_img_base = basepath + "/all_test_img"
    if not os.path.exists(new_img_base):
        os.mkdir(new_img_base)
        print("Directory " , new_img_base ,  " Created ")
    else:    
        print("Directory " , new_img_base ,  " already exists")

    # Get PNGs for each row
    all_meta["new_img_location"] = all_meta.apply(
        lambda row: restructure_file_location(
            basepath,
            new_img_base,
            row["Subject ID"],
            row["File Location"],
        ), axis=1
    )

    # Save merged and updated meta data to CSV
    all_meta.to_csv(
        "meta_data/cbis-ddsm/test_meta_with_png.csv",
        index = False,
    )

# Function to restructure a single image file location, and return the new location
def restructure_file_location(basepath, new_img_base, subject_id, old_file_location):
    
    # Get the old image path
    img_basepath = basepath + "/manifest-ZkhPvrLo5216730872708713142"
    path = pathlib.PureWindowsPath(old_file_location)
    path = img_basepath / path / "1-1.dcm"
    path = pathlib.PurePosixPath(path)

    # Get the NEW image path
    new_img_location = new_img_base + "/" + subject_id + ".png"

    # Take all DICOM images and save as 16-bit depth PNG in new folder location
    save_dicom_image_as_png(
        path, # old dicom image location
        new_img_location,  # new png image location
        bitdepth=16
    )
    
    # Add the new image locations in a new column in the meta data (caller function)
    return new_img_location

def main():
    restructure_kau_bcmd()

if __name__ == "__main__":
    main()
