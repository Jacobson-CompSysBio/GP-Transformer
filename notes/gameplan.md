# Genomic Prediction Transformer
We want to validate our methods on a known dataset

What we have:
* 4 different datasets (popular, pennycress, switchgrass, etc.)
* Basic quantitative genetics models
    1. 
* Bayesian models
    1. 
* Kernel-based models
    1. 
* ML models 
    1. iRF
    2. LightGBM
    3. SVM
* Deep Learning Models
    1. MLP-Based
        * DNN-GP
    2. CNN-Based
    3. **Transformer-Based** (Our focus)
       * DNA-BERT
       * G2P (Genotype to Phenotype)     

## Two Efforts
1. Genomic prediction
2. Genomic + climate prediction

## Training Scheme
* 5-fold CV to pick best hyperparams.
* After choosing params., train on remaining 20% of held out data

## GP-Transformer Game Plan
* **Input Data:** Genomic data
    * 51,000 SNPs for ~850 individuals in a 7x7 cross
    * Already preprocessed

* **Targets:** Four traits
    * 

* **Preprocessing:**
    * k-mer based encoding
    * BPE
    * Hard-coded vocabulary
    * other methods from literature?

* **Architecture:**
    * BERT - encode the sequences into phenotype values
        * multi-task encoding?
    * 

* **Training:**
    * Train only on 80-20 split first, no k-fold
    * maybe add k-fold to tune params. later
