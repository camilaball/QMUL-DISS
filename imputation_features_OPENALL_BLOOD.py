#importing appropriate packages 
import pandas as pd 
import numpy as np 
from functions.imputation_functions import cleaning_file, knn_imputing, merging_df, cleaning_file_merged
from functions.script_feature_selection import feature_selection
from functions.features_comparisons import bootstrap_elastic

#imputation percentages I want to test for KNN
percentages=[30,35,40,50]

#Cross validation folds I want to test on my data 
folds=[2,3,4,5]

#filepath to the open_genes cpgs and ages 
filepath='~/research_project/src/open_blood_all/not_imputed/wgbs_rrbs_merged.csv' #filepath to the cpgs related to ageing genes 
filepath2='/data/scratch/bt23801/rrbs/methylkit_tryout_blood/open_genes_model/age_runs.csv' #ages 

#looping through each percentage
for percentage in percentages: 
    data_cleaned=cleaning_file_merged(filepath,percentage)
    data_imputed=knn_imputing(5,data_cleaned)
    data_merged=merging_df(data_imputed,filepath2,percentage,'openall')
    print('done')
    
    #looping through each fold
    for fold in folds: 
        #filepath for the merged with age imputed file 
        filepath_merged=f'merged_{percentage}_openall_rrbs_only.csv'
        #running the feature selection function from publication
        feature_selection(filepath_merged,percentage,fold,'openall')
        #running bootstrapping for each and saving 
        data=pd.read_csv(filepath_merged)
        #getting the feature selection results file that was just created
        results=pd.read_csv(f'~/research_project/src/imputation_tests_openall_rrbs/feature_selection/results_all_feature_selections_{percentage}__{fold}CV.csv')
        #bootstrapping 100 times each feature selection option 16 of them as a testing method
        bootstrap_elastic(16,100,data,results,percentage,fold,'openall')
