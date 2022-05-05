import pydicom as dicom
import cv2
import matplotlib.pylab as plt

# Image filepath
base_image_path = '../data/INbreast Release 1.0/AllDICOMs'

def process_image(image_path):
    ds = dicom.dcmread(image_path)

    pixel_array_numpy = ds.pixel_array

    plt.imshow(pixel_array_numpy)

    image_format = '.jpg' # or '.png'
    image_path = "jpg" + image_path.replace('.dcm', image_format)

    cv2.imwrite(image_path, pixel_array_numpy)
    return cv2

img = process_image(base_image_path + "/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm")