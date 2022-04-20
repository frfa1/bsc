import sys, os, shutil, PIL
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
#import keras
from tensorflow import keras 

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have end2end-all-conv folder in the parent of the current folder
github_path = "../../end2end-all-conv"
sys.path.insert(1, os.path.abspath(github_path))

# Load models
#from keras.models import load_model
print("Loading models")
from tensorflow.keras.models import load_model
res_mod = load_model(github_path + '/trained_models/ddsm_resnet50_s10_[512-512-1024]x2.h5', compile=False)
vgg_mod = load_model(github_path + '/trained_models/ddsm_vgg16_s10_512x1.h5', compile=False)
hybrid_mod = load_model(github_path + '/trained_models/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5', compile=False)


# Meta df
print("Loading meta data")
#meta = pd.read_csv(
#    "../../data/cbis-ddsm/tmp_neg_pos/tmp_meta.csv"
#)
meta = pd.read_csv(
    "../bsc/meta_data/cbis-ddsm/test_meta_with_png.csv"
)
meta["true_benign"], meta["true_malignant"] = 0, 0
meta.loc[(meta["pathology"] != "MALIGNANT"), "true_benign"] = 1
meta.loc[(meta["pathology"] == "MALIGNANT"), "true_malignant"] = 1

# Data path
folder_path = os.path.abspath("../../data/cbis-ddsm/neg_pos_split")
folder_path = Path(folder_path)

## Get pixel mean
# Get list of images
"""images = []
for neg_img in os.listdir("neg"):
    images.append(os.path.join("neg/", neg_img))
for pos_img in os.listdir("pos"):
    images.append(os.path.join("pos/", pos_img))
# Get the mean
current_mean = 0
pixel_counter = 0
for img in images:
    imarr = np.array(Image.open(img),dtype=np.float)
    pixel_counter += len(imarr)
    current_mean = current_mean + (np.mean(imarr) * len(imarr))
final_mean = current_mean / pixel_counter * 0.003891"""

# Test model
print("Generating model data")
from dm_image import DMImageDataGenerator
test_imgen = DMImageDataGenerator(featurewise_center=True)
test_imgen.mean = 54.0218 #final_mean #52.18
test_generator = test_imgen.flow_from_directory(
    folder_path, target_size=(1152, 896), target_scale=None, #1152, 896
    rescale_factor = 0.003891, # For png-16
    equalize_hist=False, dup_3_channels=True, 
    classes=['neg', 'pos'], class_mode='categorical', batch_size=4, 
    shuffle=False)

## Get predictions
# Without augmentation
print("Getting predictions")
from dm_keras_ext import DMAucModelCheckpoint
res_auc, res_y_true, res_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True)
print(res_auc)

"""from dm_keras_ext import DMAucModelCheckpoint
vgg_auc, vgg_y_true, vgg_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, vgg_mod, test_samples=test_generator.nb_sample, return_y_res=True)
print(vgg_auc)

from dm_keras_ext import DMAucModelCheckpoint
hybrid_auc, hybrid_y_true, hybrid_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, hybrid_mod, test_samples=test_generator.nb_sample, return_y_res=True)
print(hybrid_auc)

from sklearn.metrics import roc_auc_score
all_mod_y_pred_avg = (res_y_pred[:,1] + vgg_y_pred[:,1] + hybrid_y_pred[:,1])/3
print(roc_auc_score(res_y_true[:,1], all_mod_y_pred_avg))

# With augmentation
from dm_keras_ext import DMAucModelCheckpoint
res_auc_aug, res_y_true_aug, res_y_pred_aug = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
print(res_auc_aug)

from dm_keras_ext import DMAucModelCheckpoint
vgg_auc_aug, vgg_y_true_aug, vgg_y_pred_aug = DMAucModelCheckpoint.calc_test_auc(
    test_generator, vgg_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
print(vgg_auc_aug)

from dm_keras_ext import DMAucModelCheckpoint
hybrid_auc_aug, hybrid_y_true_aug, hybrid_y_pred_aug = DMAucModelCheckpoint.calc_test_auc(
    test_generator, hybrid_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
print(hybrid_auc_aug)

from sklearn.metrics import roc_auc_score
all_mod_y_pred_avg_aug = (res_y_pred_aug[:,1] + vgg_y_pred_aug[:,1] + hybrid_y_pred_aug[:,1])/3
print(roc_auc_score(res_y_true_aug[:,1], all_mod_y_pred_avg_aug))"""