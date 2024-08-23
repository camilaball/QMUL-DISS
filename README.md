# QMUL-DISS
Welcome to the GitHub Repository for My Dissertation: **"Epigenetic Clocks and Age Acceleration in Cancer"**

Here you will find the following 3 folders:

- **bioinformatics_pipeline**:  
  Contains all the bash scripts used in my bioinformatics pipeline, from FastQC to methylation extraction. The final output file from this process is a Bismark coverage file.

- **methylkit_to_dataframe**:  
  Includes various R and Bash scripts for processing the Bismark coverage file, resulting in the final dataset. The dataset features sample names as columns and CpGs as rows, with methylation beta values.

- **functions**:  
  This folder contains all the Python code used for developing the age prediction model, including machine learning models, feature selection methods, etc.

Additionally, the repository includes the following files:

- **imputation_features_OPEN.py**:  
  A Python script that generates 320 different age prediction model combinations for OPEN GENES RRBS.

- **imputation_features_EPI_RRBS.py**:
   A Python script that generates 320 different age prediction model combinations for EPI GENES RRBS.
  
- **imputation_features_OPENALL_BLOOD.py**:
   A Python script that generates 256 different age prediction model combinations for OPEN BLOOD ALL.
 



