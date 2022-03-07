import pandas as pd
import sys, os
import numpy as np
import argparse
import torch
import json

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
sys.path.insert(1, os.path.abspath("../breast_cancer_classifier"))

# Github library
from src.modeling.run_model_single import (
    load_model, load_inputs, process_augment_inputs, batch_to_tensor
)
import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading

def run_model_column(row): #cropped_png_location, subject_id): #row): #df):

    global count
    count += 1
    if count % 10 == 0:
        print(count)
    
    # PNG Location and Subject ID
    #cropped_png_location, subject_id = df["cropped_png_location"], df["Subject ID"]
    cropped_png_location, subject_id = row["cropped_png_location"], row["Subject ID"]

    # Metadata file of the observation after cropping (and extracting centers)
    metadata_path = metadata_path_after_crop + "/" + subject_id + ".pkl"

    # Get the view (eg. "L-CC")
    subject_id2 = subject_id.split("_")
    view = subject_id2[-2][0] + "-" + subject_id2[-1]

    # Parameters
    shared_parameters = {
        "device_type": "cpu", #gpu
        #"gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "batch_size": 1,
        "seed": 0,
        "augmentation": True,
        "use_hdf5": True,
    }

    random_number_generator = np.random.RandomState(shared_parameters["seed"])

    image_only_parameters = shared_parameters.copy()
    image_only_parameters["view"] = view
    image_only_parameters["use_heatmaps"] = False
    image_only_parameters["model_path"] = "../breast_cancer_classifier/models/ImageOnly__ModeImage_weights.p"
    image_only_parameters["cropped_mammogram_path"] = cropped_png_location
    image_only_parameters["metadata_path"] = metadata_path
    image_only_parameters["num_epochs"] = 3

    model, device = load_model(image_only_parameters)

    model_input = load_inputs(
        image_path = image_only_parameters["cropped_mammogram_path"],
        metadata_path = image_only_parameters["metadata_path"],
        use_heatmaps = image_only_parameters["use_heatmaps"],
        #benign_heatmap_path = image_only_parameters["heatmap_path_benign"],
        #malignant_heatmap_path = image_only_parameters["heatmap_path_malignant"],
    )
    assert model_input.metadata["full_view"] == image_only_parameters["view"]

    all_predictions = []
    for data_batch in tools.partition_batch(range(image_only_parameters["num_epochs"]), image_only_parameters["batch_size"]):
        batch = []
        for _ in data_batch:
            print("BATCHING")
            batch.append(process_augment_inputs(
                model_input=model_input,
                random_number_generator=random_number_generator,
                parameters=image_only_parameters,
            ))
        tensor_batch = batch_to_tensor(batch, device)
        with torch.no_grad():
            y_hat = model(tensor_batch)
        predictions = np.exp(y_hat.cpu().detach().numpy())[:, :2, 1]
        all_predictions.append(predictions)
    agg_predictions = np.concatenate(all_predictions, axis=0).mean(0)
    predictions_dict = {
        "benign": float(agg_predictions[0]),
        "malignant": float(agg_predictions[1]),
    }
    print(json.dumps(predictions_dict))

def main():
    meta = pd.read_csv(
        "meta_data/cbis-ddsm/test_meta_after_crop.csv"
    )

    global count, metadata_path_after_crop
    count = 0
    metadata_path_after_crop = "../data/cbis-ddsm/all_test_img_output/metadata"

    # Run classifier and add prediction probabilities as new columns
    meta[["pred_benign", "pred_malignant"]] = meta[["cropped_png_location", "Subject ID"]].apply(
        run_model_column, axis=1, result_type="expand"
    )

    # Save meta data with predictions
    meta.to_csv(
        "meta_data/cbis-ddsm/test_meta_with_predictions.csv",
        index = False,
    )

if __name__ == "__main__":
    main()

