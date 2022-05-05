from email.mime import base
import png
import pydicom
import os
import pandas as pd
from pathlib import PureWindowsPath, PurePosixPath


def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())

def all_inbreast_to_png(meta_df, base_path):
    dicom_base = base_path + "/AllDICOMs"
    png_base = base_path + "/png_versions"
    
    directory = os.fsencode(dicom_base)
    for file in os.listdir(directory):

        dicom_filename = os.fsdecode(file)
        if not dicom_filename.endswith(".dcm"): # Check only for dcm files
            continue
        png_filename = os.fsdecode(file).replace(".dcm", ".png")

        full_dicom_path = dicom_base + "/" + dicom_filename
        full_png_path = png_base + "/" + png_filename

        #save_dicom_image_as_png(full_dicom_path, full_png_path)

        flnm = int(dicom_filename.split("_")[0])
        meta_df.loc[meta_df["File Name"] == flnm, "png_path"] = full_png_path
        meta_df.loc[meta_df["File Name"] == flnm, "png_filename"] = png_filename
        meta_df.loc[meta_df["File Name"] == flnm, "png_base"] = png_base

    return meta_df

def all_ddsm_to_png(meta_df, basepath):

    """ Old function to put the CBIS-DDSM PNGs to the same folder as the DICOMs """

    count = 0
    for row in meta_df.itertuples():
        count += 1
        if count % 10 == 0:
            print(count)
        
        path = PureWindowsPath(row._17)
        path = basepath / path
        path = PurePosixPath(path)

        # Create png_versions folder
        dirName = "png_versions"
        if not os.path.exists(path / dirName):
            os.mkdir(path / dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")

        for file in os.listdir(path):
            dcmpath = path / file
            if not file.endswith(".dcm"): # Check only for dcm files
                continue
            png_filename = os.fsdecode(file).replace(".dcm", ".png")
            full_png_path = path / "png_versions" / png_filename

            #print(png_filename)
            #print(dcmpath)
            #print(full_png_path)
            #print("\n")

            save_dicom_image_as_png(dcmpath, full_png_path, bitdepth=16)

            flnm = row._17
            meta_df.loc[meta_df["File Location"] == flnm, "png_path"] = full_png_path
            meta_df.loc[meta_df["File Location"] == flnm, "png_filename"] = png_filename
            meta_df.loc[meta_df["File Location"] == flnm, "png_base"] = path / "png_versions"

    return meta_df

        


def main():
    pass

    base_path = '../data/INbreast'
    meta_df = pd.read_csv(
        base_path + "/INbreast.csv",
        delimiter = ";"
    )
    meta_df["png_path"] = 0
    meta_df["png_filename"] = 0
    meta_df["png_base"] = 0
    meta_df = all_inbreast_to_png(meta_df, base_path)

    meta_data_base = "meta_data"
    meta_df.to_csv(meta_data_base + "/" "INbreast_with_img_loc.csv", index=False)



if __name__ == "__main__":
    main()