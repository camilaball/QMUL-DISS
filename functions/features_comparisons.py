import scipy
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle 
import seaborn as sns 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def bootstrap_elastic(x_range,n_testings,data,results,percentage,fold,clock):
    '''Function which performs bootstrapping over x number of iterations to compare the different feature selection methods
    x_range: each number in the range corresponds to a feature selection method, allows to iterate over different feature selection methods
    n_testings: number of iterations for bootstrapping
    data: is the dataframe containing all the cpgs and the age of each individual
    results: is the dataframe converted csv file '''
    r2_means=[] #list containing all the average r2 score over the x bootstraps for all the feature selection methods
    rmse_means=[] #list containing all the average rmse score over the x bootstraps for all the feature selection methods
    num_feats=[] #list containing the number of features for each feature selection methods
    
    method= ['boruta','rfe1500_to_boruta','rfe_1000_to_rfecv_cpgs','rfe100','sfm_elas','sfm_elas_boruta','sfm_extra','sfm_extra_boruta','rfe1500_to_sfm','kbest25', 'kbest2k_boruta','basic_elas','ga','rfe10k_ga_list','inter','inter_boruta']
    comparisons=pd.DataFrame(columns=['Method','Num_Feats','RMSE','R2'])
    comparisons['Method']=method

    y=data['Age']
    for x in range(x_range): 
        X=data.drop(['Age'],axis=1)
        X=X[results[str(x)].dropna()]
        accuracy_values=np.zeros(n_testings)
        rmses=np.zeros(n_testings)
        
        
        for i in range(n_testings):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            #fit the ElasticNetCV model
            model = ElasticNetCV(max_iter=50000, random_state=i)
            model.fit(X_train, y_train)
            #prediction on the test data
            y_pred = model.predict(X_test)
            #calculation and printing R² score
            accuracy_values[i]=r2_score(y_test,y_pred)
            rmses[i] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        average_bootstrap=np.mean(accuracy_values)
        average_rmse=np.mean(rmses)
        r2_means.append(average_bootstrap)
        rmse_means.append(average_rmse)
        num_feats.append(len(X.columns))
    comparisons['Num_Feats']=num_feats
    comparisons['RMSE']=rmse_means
    comparisons['R2']=r2_means
    comparisons.to_csv(f'imputation_tests_open_rrbs/data_{percentage}_{fold}_features.csv')
    
def bootstrap_elastic_ga(x_range,n_testings,data,results,percentage,fold,clock):
    '''Function which performs bootstrapping over x number of iterations to compare the different feature selection methods
    x_range: each number in the range corresponds to a feature selection method, allows to iterate over different feature selection methods
    n_testings: number of iterations for bootstrapping
    data: is the dataframe containing all the cpgs and the age of each individual
    results: is the dataframe converted csv file '''
    r2_means=[] #list containing all the average r2 score over the x bootstraps for all the feature selection methods
    rmse_means=[] #list containing all the average rmse score over the x bootstraps for all the feature selection methods
    num_feats=[] #list containing the number of features for each feature selection methods
    method=['Genetic Algorithm']
    comparisons=pd.DataFrame(columns=['Method','Num_Feats','R2_elastic','R2_RF'])
    comparisons['Method']=method

    y=data['Age']
    for x in range(x_range): 
        X=data.drop(['Age'],axis=1)
        X=X[results[str(x)].dropna()]
        accuracy_values=np.zeros(n_testings)
        accuracy_rf_values=np.zeros(n_testings)
        
        
        
        for i in range(n_testings):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            #fit the ElasticNetCV model
            model = ElasticNetCV(max_iter=50000, random_state=i)
            model.fit(X_train, y_train)
            #prediction on the test data
            y_pred = model.predict(X_test)
            #calculation and printing R² score
            accuracy_values[i]=r2_score(y_test,y_pred)
            model2=RandomForestRegressor(random_state=i)
        
        average_bootstrap=np.mean(accuracy_values)
        average_rmse=np.mean(rmses)
        r2_means.append(average_bootstrap)
        rmse_means.append(average_rmse)
        num_feats.append(len(X.columns))
    comparisons['Num_Feats']=num_feats
    comparisons['RMSE']=rmse_means
    comparisons['R2']=r2_means
    comparisons.to_csv(f'ga_moldels/rf_rf/data_{percentage}_{fold}_features.csv')
                  
        
        
    
                             
        
        
        
            
            
        
        
