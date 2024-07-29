#Importing the right package
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def cleaning_file(filepath: str, percentage:int): 
    '''Function which gets a csv file ready for imputation by converting 
    it to a pandas dataframe, removing unecessary columns (chrX and chrY), removing columns where there is less
    than a certain percentage of missing values, creating a new index and 
    transposing the dataframe. Returna a dataframe'''
    #importing the csv file containing the methylation levels
    df=pd.read_csv(filepath)
    #removing X and Y chromosomes from the dataframe
    df.drop(df[df['chr']=='chrX'].index,inplace=True)
    df.drop(df[df['chr']=='chrY'].index,inplace=True)
    #Creating a new index (the new index will be treated as CpG IDs) 
    df['chr_start_end']=df['chr'].astype(str)+','+df['start'].astype(str)+','+df['end'].astype(str)
    df.set_index(['chr_start_end'],inplace=True)
    df.drop(columns=['chr','start','end','gene_symbol'],inplace=True)
    #columns where there is less thna 30% of data missing 
    #total number of missing values 
    ageing=df.transpose()
    nan_counts=ageing.isna().sum()
    total_counts=ageing.shape[0]
    #percentage of Nan values in each columns
    nan_percentage=nan_counts/ageing.shape[0]*100
    columns_less_nans=nan_percentage[nan_percentage<percentage]
    #dataframe with the index of all the cpgs where there is less than 30% of data missing
    less_nans=pd.DataFrame(columns_less_nans)
    #Creating new df which has less than 20% of missing values 
    df_30=pd.merge(less_nans,df,left_index=True,right_index=True)
    df_30.drop(columns=[0],inplace=True)
    df30=df_30.transpose()
    return(df30) 

def knn_imputing(value:int,df):
    '''Function which imputes missing values using the k-nearest neighbor algorithm.
    This functionshould be used after cleaning_file and df is the dataframe returned from cleaning_file.'''
    knn_imputer=KNNImputer(n_neighbors=value)
    imputed_data=knn_imputer.fit_transform(df)
    imputed_df=pd.DataFrame(imputed_data,columns=df.columns,index=df.index)
    #imputed_df.to_csv('imputed_20_epi_rrbs_only.csv')
    return(imputed_df) 


def merging_df(imputed_df,filepath:str,percentage:int,clock:str):
    '''Function which adds the age column to the imputed dataframe for the subsequent steps.
    This function should only be used after knn imputing function'''
    df_age=pd.read_csv(filepath,index_col=0)
    actual_age=pd.merge(imputed_df,df_age,left_index=True,right_index=True)
    actual_age=actual_age[['Age']]
    merged_df=pd.merge(imputed_df,actual_age,left_index=True,right_index=True)
    merged_df.to_csv(f'merged_{percentage}_{clock}_rrbs_only.csv',index=False)
    
    return(merged_df)
    
    


    
    
   
    
