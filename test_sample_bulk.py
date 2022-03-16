import pandas as pd
import sys, os
import numpy as np

# Inserting siebling folder to sys. Ensures other code can be ran
# Note: Must have breast_cancer_classifier folder in the parent of the current folder
sys.path.insert(1, os.path.abspath("../breast_cancer_classifier"))

# Github library
#import src.modeling.run_model_single as run_single
from src.modeling.run_model_single import (
    load_model, load_inputs, process_augment_inputs, batch_to_tensor
)
import src.cropping.crop_single as crop_single
import src.optimal_centers.get_optimal_center_single as center_single
import src.utilities.pickling as pickling
import src.utilities.tools as tools


def run(
    img_location,
    cropped_mammogram_path,
    metadata_path,
    view
    ):

    crop_single.crop_single_mammogram(
        mammogram_path = img_location,
        horizontal_flip = "NO",
        view = view,
        cropped_mammogram_path = cropped_mammogram_path,
        metadata_path = metadata_path,
        num_iterations = 100,
        buffer_size = 50
    )

    center_single.get_optimal_center_single(
        cropped_mammogram_path = cropped_mammogram_path,
        metadata_path = metadata_path
    )

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

    model, device = load_model(image_only_parameters)
    model_input = load_inputs(
        image_path = cropped_mammogram_path,
        metadata_path = metadata_path,
        use_heatmaps=False,
    )
    batch = [
        process_augment_inputs(
            model_input=model_input,
            random_number_generator=random_number_generator,
            parameters=image_only_parameters,
        ),
    ]
    tensor_batch = batch_to_tensor(batch, device)
    y_hat = model(tensor_batch)
    predictions = np.exp(y_hat.cpu().detach().numpy())[:, :2, 1]
    predictions_dict = {
        "benign": float(predictions[0][0]),
        "malignant": float(predictions[0][1]),
    }
    return_value = [
        predictions_dict["benign"],
        predictions_dict["malignant"]
    ]
    print(return_value)


def main():

    img_location = "../breast_cancer_classifier/sample_data/images/0_L_CC.png"
    cropped_mammogram_path = "../breast_cancer_classifier/sample_data/images/cropped/0_L_CC.png"
    metadata_path = "../breast_cancer_classifier/sample_data/images/cropped/0_L_CC.pkl"
    view = "L-CC"

    run(
        img_location,
        cropped_mammogram_path,
        metadata_path,
        view
    )

if __name__ == "__main__":
    main()