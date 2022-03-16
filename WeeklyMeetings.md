# Weekly Meetings

## 10-02-2022

I found data set / model candidates, and we talked about an approach. Next action: Train an existing model and test it's resistance to hidden features.

## 17-02-2022 - 24-02-2022

We've talked about different datasets and models based on research papers in the field of mammograms. I've tested a particular model on CBIS dataset with poor results.

## 17-03-2022

I suggest we talk about the following:

- What should I use as true labels for the CBIS data set?
- - Rows correspond to segmentation labels, not for whole images)
- - Further, the rows contain both biopsy pathology and radiologist assessment, and they produce slightly difference AOC results. The paper mentions that >30% of the cases in their own data are not present on mammograms but only found through biopsy.
- How can I ensure that the normalisation / preprocessing methods in the pipeline are correct when testing? I.e. what are the exact steps I should look for?
- Which methods should I use to analyze differences across data sets, that may account for lack of model robustness?
- Another resort is to give the model the optimal cicumstancens by adding "heatmap generation" step to the pipeline and do the test on cluster (GPU).