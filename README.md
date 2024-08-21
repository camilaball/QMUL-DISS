Welcome to the GitHub repository of my dissertation titled 'Epigenetic Clocks and Age Acceleration in Cancer" 

Here there are 3 folders: 
- bioinformatics_pipeline: Folder contains all the bash script I used for my bioinformatics pipeline from FastQC to methylation extraction where the final output file was a Bismark coverage file
- methylkit_to_dataframe: Folder contains the different R and Bash scripts for processing the Bismark coverage file until getting the final dataset containing as columns hte sample names and the cpgs as rows (with the values being the methylation beta values)
- functions : python folder containing all the python code for the development of age prediction model (machine learning models, feature selection methods etc.)

  The 2 other files are :
  - imputation_features_OPEN.py: python script generating the 320 age prediction model combinations for OPEN GENES RRBS
  - imputation_features_EPI.py: python script generating the 320 age prediction model combinations for EPI GENES RRBS
