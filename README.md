# Finding Hidden Features Responsible For Machine Learning Failures In Breast Cancer Detection on Screening Mammography

Bachelor Project, May 2022.

## Prerequisite

To run this repository, a "/data" folder with the INbreast, CBIS-DDSM and potentially kau-bcmd data sets must be placed locally as siblings folders relative to the repository. Similarly, the [end2end repository](https://github.com/lishen/end2end-all-conv) must be placed at the same level locally as this repository. This way, data and end2end models are fetched with relatived paths, i.e.: "../data/CBIS-DDSM/...".

## Folder structure

**/data_exploration**: This folder exemplifies some data exploration of CBIS-DDSM and INbreast. "Heatmap generation.ipynb" contains code for patch classifier heatmaps, that utilizes functionality in the end2end Github.

**/load_preprocess** specifies functionality to load data sets from relative paths, and to transform data, i.e. from DICOM to PNG-16.

**/model1**: Here are the initial workings with the [secondary research models](https://github.com/nyukat/breast_cancer_classifier), which were later discarded due to private data set.

**/model2**: Here are the workings with the end2end model Github. For example, test_cbis.py tests all the models on CBIS-DDSM test, and outputs instance-level posterior probabilities to later compute AUROCs. run_test_cbis.job is an example script ran on the ITU cluster, by setting up environment and calling test_cbis.py. Special prerequisites here are: Keras <= 2.0.8 and Tensorflow < 2.0.0.

**/results_analysis** contains functionality that extracts and computes results from the model outputs. For instance, get_results.py computes model-level AUROCS on both CBIS-DDSM and INbreast.




