import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

# Inserting parent folder in sys, to allow imports
sys.path.append("..")
from load_preprocess.load_meta import get_cbis_test
from load_preprocess.load_meta import get_inbreast

def get_auc(true_malignant, pred_malignant, save_roc_as=None):

    # Get ROC Curve
    fpr, tpr, _ = metrics.roc_curve(
        true_malignant, # y_true
        pred_malignant # y_score
    )
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

def get_raidiolgist_auc(meta, feature=None, save=False):

    roi_meta = get_cbis_test(whole_image_labels=False) # Get full roi-level meta with descriptive columns

    # Assign birads >= 4 to malignant
    roi_meta["radio_malignant"] = 0
    roi_meta.loc[(roi_meta["assessment"] >= 4), "radio_malignant"] = 1

    # Ensures ambiguous ROIs are labeled malignant, if any ROI is malignant
    true_cases = roi_meta[roi_meta["radio_malignant"] == 1]["Subject ID"]
    roi_meta.loc[roi_meta["Subject ID"].isin(true_cases), "radio_malignant"] = 1

    roi_meta.drop_duplicates(
        subset=["Subject ID"], inplace=True
    ) # 645 Subject IDs with whole image labels.
    meta["Subject ID"] = meta["Filename"].str.split("/").str[1].str.split(".").str[0]
    full_meta = roi_meta.merge(meta, on="Subject ID") # Join to get labels

    # Join features
    meta_features = pd.read_csv(
        "../../data/cbis-ddsm/meta/cbis_test_with_features.csv"
    ) # Get meta with features
    full_meta = full_meta.merge(meta_features, on="Subject ID")

    # Attempt to skip birads 3
    #full_meta = full_meta[full_meta["assessment"] != 3]

    # Subset excluding the feature
    if feature:
        full_meta = full_meta[full_meta[feature] != 1]

    auc = get_auc(
        full_meta.loc[:,"true_pos"], # True
        full_meta.loc[:,"radio_malignant"] # Pred
    )

    print("Radiologist AUC excluding feature:", feature)
    print(auc)

    """if save:
        auc_df.to_excel(
            "../../data/cbis-ddsm/results/ablation/" + save,
            index = False
        )"""

def inbreast_test_end2end_auc(save=False):
    meta_pred = pd.read_csv(
        "../../data/INbreast/results/end2end_inbreast_test_results.csv"
    )

    model_names = [
        'res_pred_pos',
       'vgg_pred_pos',
       'hybrid_pred_pos',
       'res_pred_pos_aug',
       'vgg_pred_pos_aug',
       'hybrid_pred_pos_aug'
    ]

    for model in model_names:
        print("Getting AUC for", model)
        print(
            get_auc(
                meta_pred["true_pos"], # True labels
                meta_pred[model]
            )
        )


def cbis_test_end2end_auc(meta, feature=None, ablation=True, save=False):

    """ Gets AUROC on birads level per model, including averages """

    roi_meta = get_cbis_test(whole_image_labels=False) # Get full roi-level meta with descriptive columns
    meta["Subject ID"] = meta["Filename"].str.split("/").str[1].str.split(".").str[0] # Craft Subject ID column in image-level meta

    meta_features = pd.read_csv(
        "../../data/cbis-ddsm/meta/cbis_test_with_features.csv"
    ) # Get meta with features
    meta_features = meta.merge(meta_features, on="Subject ID")

    full_meta = meta_features.merge(roi_meta, on="Subject ID")
    birads = sorted(list(full_meta["assessment"].unique())) # Get all unique birad assessments 

    # Subset by feature
    if feature:
        if ablation:
            full_meta = full_meta[full_meta[feature] != 1]
        else:
            full_meta = full_meta[full_meta[feature] == 1]

    print("Full meta")
    print(full_meta)


    for birad in birads: # Iterate through all available BIRADs
        # Subset the BIRAD
        if birad == 1: # Merge birad 1 & 2
            tmp_meta = full_meta[(full_meta["assessment"] == 1) | (full_meta["assessment"] == 2)]
            birad_ = "1 & 2"
        elif birad == 2: # And skip birad 2
            continue
        else:
            tmp_meta = full_meta[full_meta["assessment"] == birad]
            birad_ = birad
        # Then getting counts of birads at ROI-level
        n_pos = tmp_meta["true_pos"].sum()
        n_neg = tmp_meta["true_neg"].sum()

        # Then, remove duplicates to only report whole image classification
        tmp_meta.drop_duplicates(
            subset=["Subject ID"], inplace=False
        )
        
        print("--- Birads", birad_, "---")
        print(tmp_meta)

        if tmp_meta.empty:
            continue

        pred_pos_list = [
            # Normal models
            tmp_meta.loc[:,"res_pred_pos"],
            tmp_meta.loc[:,"vgg_pred_pos"],
            tmp_meta.loc[:,"hybrid_pred_pos"],
            (tmp_meta.loc[:,"res_pred_pos"] + tmp_meta.loc[:,"vgg_pred_pos"] + tmp_meta.loc[:,"hybrid_pred_pos"]) / 3, # Average

            # With augmentation
            tmp_meta.loc[:,"res_pred_pos_aug"],
            tmp_meta.loc[:,"vgg_pred_pos_aug"],
            tmp_meta.loc[:,"hybrid_pred_pos_aug"],
            (tmp_meta.loc[:,"res_pred_pos_aug"] + tmp_meta.loc[:,"vgg_pred_pos_aug"] + tmp_meta.loc[:,"hybrid_pred_pos_aug"]) / 3 # Average
        ]
        auc_list = []
        for pred_pos in pred_pos_list:
            tmp_auc = get_auc(
                tmp_meta.loc[:,"true_pos"],
                pred_pos
            )
            auc_list.append(tmp_auc) # Append AUC

        tmp_auc_df = pd.DataFrame(
            data = {
                "birads": birad_,
                "n_pos": n_pos,
                "n_neg": n_neg,

                "res_auc": [auc_list[0]],
                "vgg_auc": [auc_list[1]],
                "hybrid_auc": [auc_list[2]],
                "avg_auc": [auc_list[3]],

                "res_aug_auc": [auc_list[4]],
                "vgg_aug_auc": [auc_list[5]],
                "hybrid_aug_auc": [auc_list[6]],
                "avg_aug_auc": [auc_list[7]]
            }
        )

        if not "auc_df" in locals():
            auc_df = tmp_auc_df
        else:
            auc_df = pd.concat([auc_df, tmp_auc_df], ignore_index=True)

    # Get AUC for all BIRADs
    meta = full_meta.drop_duplicates(
            subset=["Subject ID"]
        )
    print(meta)
    pred_pos_list = [
        # Normal models
        meta.loc[:,"res_pred_pos"],
        meta.loc[:,"vgg_pred_pos"],
        meta.loc[:,"hybrid_pred_pos"],
        (meta.loc[:,"res_pred_pos"] + meta.loc[:,"vgg_pred_pos"] + meta.loc[:,"hybrid_pred_pos"]) / 3, # Average
        # With augmentation
        meta.loc[:,"res_pred_pos_aug"],
        meta.loc[:,"vgg_pred_pos_aug"],
        meta.loc[:,"hybrid_pred_pos_aug"],
        (meta.loc[:,"res_pred_pos_aug"] + meta.loc[:,"vgg_pred_pos_aug"] + meta.loc[:,"hybrid_pred_pos_aug"]) / 3 # Average
    ]
    all_auc_list = []
    for pred_pos in pred_pos_list:
        all_auc = get_auc(
            meta.loc[:,"true_pos"],
            pred_pos
        )
        all_auc_list.append(all_auc) # Append AUC
    all_auc_df = pd.DataFrame(
        data = {
            "birads": "All",
            "n_pos": meta["true_pos"].sum(),
            "n_neg": meta["true_neg"].sum(),
            "res_auc": [all_auc_list[0]],
            "vgg_auc": [all_auc_list[1]],
            "hybrid_auc": [all_auc_list[2]],
            "avg_auc": [all_auc_list[3]],
            "res_aug_auc": [all_auc_list[4]],
            "vgg_aug_auc": [all_auc_list[5]],
            "hybrid_aug_auc": [all_auc_list[6]],
            "avg_aug_auc": [all_auc_list[7]]
        }
    )
    auc_df = pd.concat([all_auc_df, auc_df], ignore_index=True)

    if save:
        auc_df.to_excel(
            "../../data/cbis-ddsm/results/ablation/" + save,
            index = False
        )

    return auc_df


def main():
    """meta = pd.read_csv(
        "../../data/cbis-ddsm/results/end2end_cbis_test_results.csv"
    )
    features = [
        ("feature_text", "feature_text_ablation.xlsx"),
        ("feature_nipple_dot", "feature_nipple_dot_ablation.xlsx"),
        ("feature_scar_line", "feature_scar_line_ablation.xlsx"),
        ("feature_ruler", "feature_ruler_ablation.xlsx")
    ]
    for feature in features:
        print("\n", feature)
        auc_df = cbis_test_end2end_auc(meta, feature=feature[0], ablation=True, save=feature[1])
    auc_df = cbis_test_end2end_auc(meta, save="no_ablation.xlsx")

    for feature in features:
        get_raidiolgist_auc(meta, feature=feature[0])
    get_raidiolgist_auc(meta)"""

    inbreast_test_end2end_auc()

if __name__ == "__main__":
    main()