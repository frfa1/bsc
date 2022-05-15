

print("Loading libraries")
import sys, os, shutil, PIL

# Avoid running on GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from functools import reduce
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import keras

# Following block is extra
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

#import tensorflow as tf
print("Libraries loaded")

#exit() # For troubleshooting -- Error happens just on keras/tensorflow import, when GPU is available /Frederik

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have end2end-all-conv folder in the parent of the current folder
github_path = "../../end2end-all-conv"
sys.path.insert(1, os.path.abspath(github_path))

# Load models
print("Loading models")
from keras.models import load_model
#from tensorflow.keras.models import load_model
res_mod = load_model(github_path + '/trained_models/ddsm_resnet50_s10_[512-512-1024]x2.h5', compile=False)
vgg_mod = load_model(github_path + '/trained_models/ddsm_vgg16_s10_512x1.h5', compile=False)
hybrid_mod = load_model(github_path + '/trained_models/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5', compile=False)

# Data path
folder_path = os.path.abspath("../../data/INbreast/test")
folder_path = Path(folder_path)

## Get pixel mean
# Get list of images
images = []
for neg_img in os.listdir(folder_path / "neg"):
    images.append(os.path.join(folder_path / "neg/", neg_img))
for pos_img in os.listdir(folder_path / "pos"):
    images.append(os.path.join(folder_path / "pos/", pos_img))
# Get the mean
current_mean = 0
pixel_counter = 0
for img in images:
    imarr = np.array(Image.open(img),dtype=np.float)
    pixel_counter += len(imarr)
    current_mean = current_mean + (np.mean(imarr) * len(imarr))
final_mean = current_mean / pixel_counter * 0.003891

print("Data input pixel mean", final_mean)

# Test model
print("Generating model data")
from dm_image import DMImageDataGenerator
test_imgen = DMImageDataGenerator(featurewise_center=True)
test_imgen.mean = final_mean #54.0218 #final_mean #52.18
test_generator = test_imgen.flow_from_directory(
    folder_path, target_size=(1152, 896), target_scale=None, #1152, 896
    rescale_factor = 0.003891, # For png-16
    equalize_hist=False, dup_3_channels=True, 
    classes=['neg', 'pos'], class_mode='categorical', batch_size=4, 
    shuffle=False)
print("Test generator", dir(test_generator))

## Get predictions
# Without augmentation
print("Getting predictions")

from dm_keras_ext import DMAucModelCheckpoint
res_auc, res_y_true, res_y_pred, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True)
res_df = pd.DataFrame(
    data = {
        "Filename": filenames_list, 
        "true_neg": res_y_true[:,0],
        "true_pos": res_y_true[:,1],
        "res_pred_neg": res_y_pred[:,0], 
        "res_pred_pos": res_y_pred[:,1]
    }
)

from dm_keras_ext import DMAucModelCheckpoint
vgg_auc, vgg_y_true, vgg_y_pred, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, vgg_mod, test_samples=test_generator.nb_sample, return_y_res=True)
vgg_df = pd.DataFrame(
    data = {
        "Filename": filenames_list, 
        "vgg_pred_neg": vgg_y_pred[:,0], 
        "vgg_pred_pos": vgg_y_pred[:,1]
    }
)

from dm_keras_ext import DMAucModelCheckpoint
hybrid_auc, hybrid_y_true, hybrid_y_pred, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, hybrid_mod, test_samples=test_generator.nb_sample, return_y_res=True)
hybrid_df = pd.DataFrame(
    data = {
        "Filename": filenames_list, 
        "hybrid_pred_neg": hybrid_y_pred[:,0], 
        "hybrid_pred_pos": hybrid_y_pred[:,1]
    }
)

### With augmentation ###
from dm_keras_ext import DMAucModelCheckpoint
res_auc_aug, res_y_true_aug, res_y_pred_aug, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
res_df_aug = pd.DataFrame(
    data = {
        "Filename": filenames_list,
        "res_pred_neg_aug": res_y_pred_aug[:,0], 
        "res_pred_pos_aug": res_y_pred_aug[:,1]
    }
)

from dm_keras_ext import DMAucModelCheckpoint
vgg_auc_aug, vgg_y_true_aug, vgg_y_pred_aug, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, vgg_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
vgg_df_aug = pd.DataFrame(
    data = {
        "Filename": filenames_list, 
        "vgg_pred_neg_aug": vgg_y_pred_aug[:,0], 
        "vgg_pred_pos_aug": vgg_y_pred_aug[:,1]
    }
)

from dm_keras_ext import DMAucModelCheckpoint
hybrid_auc_aug, hybrid_y_true_aug, hybrid_y_pred_aug, filenames_list = DMAucModelCheckpoint.calc_test_auc(
    test_generator, hybrid_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
hybrid_df_aug = pd.DataFrame(
    data = {
        "Filename": filenames_list, 
        "hybrid_pred_neg_aug": hybrid_y_pred_aug[:,0], 
        "hybrid_pred_pos_aug": hybrid_y_pred_aug[:,1]
    }
)

data_frames = [
    res_df,
    vgg_df,
    hybrid_df,
    res_df_aug,
    vgg_df_aug,
    hybrid_df_aug
]

df_merged = reduce(
    lambda  left,right: pd.merge(left,right,on=['Filename'], how='outer'), data_frames
    )

print("\n")
print("--- SCRIPT FINISHED! ---")
print(df_merged)


# Save merged and updated meta data to CSV
results_folder = "../../data/INbreast" + "/results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
df_merged.to_csv(
    results_folder + "/end2end_inbreast_test_results.csv",
    index = False,
)



## Get averaged AUCs
"""from sklearn.metrics import roc_auc_score
all_mod_y_pred_avg = (res_y_pred[:,1] + vgg_y_pred[:,1] + hybrid_y_pred[:,1])/3
print(roc_auc_score(res_y_true[:,1], all_mod_y_pred_avg))

from sklearn.metrics import roc_auc_score
all_mod_y_pred_avg_aug = (res_y_pred_aug[:,1] + vgg_y_pred_aug[:,1] + hybrid_y_pred_aug[:,1])/3
print(roc_auc_score(res_y_true_aug[:,1], all_mod_y_pred_avg_aug))"""