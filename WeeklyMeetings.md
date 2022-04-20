# Weekly Meetings

## 10-02-2022

I found data set / model candidates, and we talked about an approach. Next action: Train an existing model and test it's resistance to hidden features.

## 17-02-2022 - 24-02-2022

We've talked about different datasets and models based on research papers in the field of mammograms. I've tested a particular model on CBIS dataset with poor results.

## 17-03-2022

I suggest we talk about the following:

**Overall goal: Define the steps needed before I can trust the results of model tests**
- What should I use as true labels in the CBIS data set?
  - Rows correspond to segmentation labels, not whole images
  - Further, the rows contain both **biopsy pathology** and **radiologist assessment**, and they produce slightly difference AUC results. The paper mentions that >30% of the cases in their own data are not present on mammograms but only found through biopsy.
- How can I ensure that the normalisation / preprocessing methods in the pipeline are correct when testing? I.e. what are the exact steps I should look for?
- Which methods should I use to analyze the differences across data sets, that may account for a lack of model robustness?
- Another resort is to give the model the optimal cicumstancens by adding the "heatmap generation" step to the pipeline and do the test on cluster (GPU).

**Edit / own suggestions**:
- E-mail the founders of the model/paper and ask for access to a subset of their data to test on, as a way to overfit
- Train the model from scratch on a varied data set, including subsets of the data to test on. This ensures consistent data distributions.

## 24-03-2022
**What did you achieve?**
- Found an existing model and got better than random AUC results on CBIS-DDSM test (77%)

**What did you struggle with?**
- Might need a plan for using the model in the context of hidden features.
  - I.e. is 77% good enough results to do error analysis on the cases?
  - The model Github has tools to apply transfer learning on a new data set, which may be beneficial to “fix” errors on hidden features. But I need a good strategy first

**What would you like to work on next week?**
- Test the model on INbreast also, which I hypothesize will yield worse results, as the model is trained on CBIS (and therefore only knows that data distribution). This can “prove” lack of model robustness on mammography scanners and image preprocessing (aka hidden features)
- Try transfer learning on the model with INbreast

**Where do you need help from Veronika?**
- Inputs to transfer learning strategy
- Any inputs to the above

**Conclusion / Next steps**
- Find which subsets of the test sets have hidden features, e.g. pacemaker
- Unbiasing / debiasing / fairness-aware methods

## 06-04-2022
**What did you achieve?**
- Looked through all CBIS and INBreast images in search for hidden features
- Found another mammography dataset (from Saudi Arabia) with multiple instances of implants
- Started working on the cluster 

**What did you struggle with?**
- Finding relevant hidden features in CBIS and INBreast
  - In all of CBIS (train+test), I only found 1 pacemaker, 0 implants, some cases with an unnaturally "even" circle, and a good bunch of cases with text
- Setting up a virtualenv on cluster, which is required to run the external Github (waiting answer from support)

**What would you like to work on next week?**
- Since the newly found dataset contains implants instances, I would like to try a couple of things on the cluster:
  - Test the model (which has been trained on CBIS) on this dataset
  - Transfer learn the model on the new data with different splits (one split where instances of implants are only present in the test set, and one where instances are present in both train and test set). This way, I check if the model struggles on cases of implants, and if it is fixable through including these types of cases in train.

**Where do you need help from Veronika?**
- Any inputs to the above

## 20-04-2022
**What did you achieve?**
- Labeled all CBIS official test test throughouly
- Wrote script to temporarily put images with a certain feature / BIRAD into a temporary folder, to test them on the model (the model expects a folder of images as input)
  - Then realised it's smarter simply to run a single experiement, and segment results after
- I began writing the report and have described the CBIS dataset

**What did you struggle with?**
- The final processing decision of the data with regards to group ambigious cases.
- I need to find a way to map the input test cases/images to the output posterior probabilities from the model
- Tensorflow 1.15 still not working with GPU on HPC Cluster. Still in dialogue with Lottie after Easter break, who has just installed cuda10.0 and cudnn7.4 on cluster - still no luck.

**What would you like to work on next week?**
- The points above, especially 1 and especially 2, to get the results
- If bottlenecks occur, I will work on the report or duplicate the text images without the text

**Where do you need help from Veronika?**
- Any inputs to the above


