import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_roc_curve(meta):

    # Assign actual pathology to 1 or 0 for both classes (benign and malignant)
    meta["true_benign"], meta["true_malignant"] = 0, 0
    meta.loc[(meta["pathology"] != "MALIGNANT"), "true_benign"] = 1
    meta.loc[(meta["pathology"] == "MALIGNANT"), "true_malignant"] = 1

    # Normalised prediction probabilities
    meta["pred_malignant_norm"] = meta["pred_malignant"] / (meta["pred_malignant"] + meta["pred_benign"])
    meta["pred_benign_norm"] = meta["pred_benign"] / (meta["pred_malignant"] + meta["pred_benign"])
    
    """meta["norm_plus"] = meta["pred_malignant_norm"] + meta["pred_benign_norm"]
    print(meta[[
        "pred_malignant",
        "pred_benign",
        "pred_malignant_norm",
        "pred_benign_norm",
        "norm_plus"
        ]])"""

    # Get malignant ROC Curve
    fpr, tpr, _ = metrics.roc_curve(
        meta[["true_malignant"]], # y_true
        meta[["pred_malignant_norm"]] # y_score
    )
    roc_auc = metrics.auc(fpr, tpr)

    # Plotting ROC Curve
    plt.plot(
        [0, 1], [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
    )
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f, n = %0.0i)" % (roc_auc, len(meta))
    )
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve - Malignant")
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(
        "meta_data/cbis-ddsm/malignant_roc_curve.png",
        )

    print(
        meta\
            .sort_values("pred_malignant_norm")
    )

    """print(
        meta[meta["true_malignant"] == 1]
    )"""

    """print(
        list(
            meta.sort_values("pred_malignant_norm")["left or right breast"]
        )
    )"""

    """print(
        meta[meta["abnormality type"] == "mass"]\
            .sort_values("pred_malignant_norm")
    )
    print(
        meta[meta["abnormality type"] == "calcification"]\
            .sort_values("pred_malignant_norm")
    )"""

    #print(meta[["abnormality type"]])

    return

def main():
    meta = pd.read_csv(
        "meta_data/cbis-ddsm/test_meta_with_predictions.csv"
    )
    get_roc_curve(meta)

if __name__ == "__main__":
    main()