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
from scipy import stats




def creating_df_cpgs(filepath:str):
    '''Function which returns a list containing all the cpgs that were feature selected'''
    df=pd.read_csv(filepath)
    cpgs=list(df['0'])
    return(cpgs)

def ElasticNet(cpgs:list,merged_df,test_size:float,random_state:int):
    '''Function which trains an ElasticNet model on the selected CpGs and 
    returns y_test and y_pred and the r2 score'''
    #data
    input_data = merged_df
    X = input_data[cpgs]  #list of all the features
    y=input_data['Age']
    #data into training and testing sets consistently
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    #fit the ElasticNetCV model
    model = ElasticNetCV(max_iter=50000, random_state=random_state)
    model.fit(X_train, y_train)
    #prediction on the test data
    y_pred = model.predict(X_test)
    #calculation and printing R² score
    r2 = r2_score(y_test, y_pred)
    print(f"R² score: {r2}")
    # Calculate and print RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n_cv = 5
    r2_scores = cross_val_score(model, X, y, cv=n_cv, scoring='r2')
    #calculate the pearson coefficient
    pearson_r = scipy.stats.pearsonr(y_test, y_pred)

# Calculate the mean R² score
    average_r2 = np.mean(abs(r2_scores))
    print(f'R² scores for each fold: {r2_scores}')
    print(f'Average R² score across {n_cv} folds: {average_r2}')
    print(f"RMSE: {rmse}")
    print(f"Pearson correlation: {pearson_r}")
    return(y_test,y_pred,r2,pearson_r[0])
               
def graph_ytest_ypred(y_test,y_pred,r2,pearson_r):
    '''Function which creates a graph demonstrating the relationship between actual age and predicted age'''
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Actual vs Predicted Age (R² = {r2:.2f}, R = {pearson_r:.2f})')
    plt.grid(True)
    plt.savefig('complete_graph_epi_rrbs.png')

    plt.show()
    
def graph_pearson_ytest_ypred(y_test,y_pred,pearson_r):
    '''Function which creates a graph demonstrating the relationship between actual age and predicted age'''
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Actual vs Predicted Age (Pearson correlation = {pearson_r:.2f})')
    plt.grid(True)
    plt.savefig('pearson_correlaiton_graph_EPI_rrbs.png')

    plt.show()
               
def bootstrap_permutation_elastic(merged_df,cpgs):
    '''Function that performs bootstrapping and permutation testing using ElasticNet model'''
    n_testings=100
    #data
    input_data = merged_df
    X = input_data[cpgs]  #list of all the features
    y = input_data['Age']

    #arrays to store all the relevant accuracy datasets 
    accuracy_values=np.zeros(n_testings)
    permuted_accuracy_values=np.zeros(n_testings)

    #bootstrapping 100 times + data into training and testing sets consistently
    for i in range(n_testings):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        #fit the ElasticNetCV model
        model = ElasticNetCV(max_iter=50000, random_state=i)
        model.fit(X_train, y_train)
        #prediction on the test data
        y_pred = model.predict(X_test)
        #calculation and printing R² score
        accuracy_values[i]=r2_score(y_test,y_pred)

    average_bootstrap=np.mean(accuracy_values)

    for i in range(n_testings):
        y_permuted=shuffle(y,random_state=i)
        X_train_permuted, X_test_permuted, y_train_permuted, y_test_permuted = train_test_split(X, y_permuted, test_size=0.2,random_state=i)
        #fit the ElasticNetCV model
        model = ElasticNetCV(max_iter=50000, random_state=i)
        model.fit(X_train_permuted, y_train_permuted)
        #prediction on the test data
        y_pred_permuted = model.predict(X_test_permuted)
        #calculation and printing R² score
        permuted_accuracy_values[i]=r2_score(y_test_permuted,y_pred_permuted)

    average_permuted_accuracy=np.mean(permuted_accuracy_values)
    ttest=stats.ttest_ind(accuracy_values,permuted_accuracy_values)
    plt.figure(figsize=(14, 10))
    sns.kdeplot(accuracy_values,label='Orginal data (non-permuted)',
           fill=True, common_norm=False, color='blue')
    sns.kdeplot(permuted_accuracy_values,color='red',label='Permuted Data',
           fill=True,common_norm=False)
    

    plt.axvline(average_bootstrap, color='blue',linestyle='dashed', linewidth=2, label='Avg r2 score non permuted')
    plt.axvline(average_permuted_accuracy, color='red',linestyle='dashed',linewidth=2, label='Avg r2 score permuted')
 
    plt.legend()
    plt.text(average_bootstrap, plt.ylim()[1]*0.5, f'{average_bootstrap:.2f}', color='blue', fontsize=26, ha='right')
    plt.text(average_permuted_accuracy, plt.ylim()[1]*0.5, f'{average_permuted_accuracy:.2f}', color='red', fontsize=26, ha='right')
    plt.xlabel('R2 Score')
    plt.title('Comparison of Original and Permuted Data R2 Scores')
    plt.savefig('Permuted_Original_Graphs_EPI_RRBS.png')
    
    return(average_bootstrap,average_permuted_accuracy,accuracy_values,ttest)


def bootstrap_pearson_permutation_elastic(merged_df,cpgs):
    '''Function that performs bootstrapping and permutation testing using ElasticNet model'''
    n_testings=100
    #data
    input_data = merged_df
    X = input_data[cpgs]  #list of all the features
    y = input_data['Age']

    #arrays to store all the relevant accuracy datasets 
    accuracy_values=np.zeros(n_testings)
    permuted_accuracy_values=np.zeros(n_testings)

    #bootstrapping 100 times + data into training and testing sets consistently
    for i in range(n_testings):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        #fit the ElasticNetCV model
        model = ElasticNetCV(max_iter=50000, random_state=i)
        model.fit(X_train, y_train)
        #prediction on the test data
        y_pred = model.predict(X_test)
        #calculation and printing R² score
        accuracy_values[i]=scipy.stats.pearsonr(y_test, y_pred)[0]

    average_bootstrap=np.mean(accuracy_values)

    for i in range(n_testings):
        y_permuted=shuffle(y,random_state=i)
        X_train_permuted, X_test_permuted, y_train_permuted, y_test_permuted = train_test_split(X, y_permuted, test_size=0.2,random_state=i)
        #fit the ElasticNetCV model
        model = ElasticNetCV(max_iter=50000, random_state=i)
        model.fit(X_train_permuted, y_train_permuted)
        #prediction on the test data
        y_pred_permuted = model.predict(X_test_permuted)
        #calculation and printing R² score
        permuted_accuracy_values[i]=scipy.stats.pearsonr(y_test_permuted,y_pred_permuted)[0]

    average_permuted_accuracy=np.mean(permuted_accuracy_values)
    plt.figure(figsize=(14, 10))
    sns.kdeplot(accuracy_values,label='Orginal data (non-permuted)',
           fill=True, common_norm=False, color='blue')
#     sns.kdeplot(permuted_accuracy_values,color='red',label='Permuted Data',
#            fill=True,common_norm=False)
    

    plt.axvline(average_bootstrap, color='blue',linestyle='dashed', linewidth=2, label='Avg r score non permuted')
#     plt.axvline(average_permuted_accuracy, color='red',linestyle='dashed',linewidth=2, label='Avg r score permuted')
 
    plt.legend()
    plt.text(average_bootstrap, plt.ylim()[1]*0.5, f'{average_bootstrap:.2f}', color='blue', fontsize=26, ha='right')
#     plt.text(average_permuted_accuracy, plt.ylim()[1]*0.5, f'{average_permuted_accuracy:.2f}', color='red', fontsize=12, ha='right')
    plt.xlabel('R Score')
    plt.title('Pearson correlation coefficient over 100 bootstraps')
    plt.savefig('R_bootrap_epi_genes_rrbs.png')
    return(average_bootstrap,average_permuted_accuracy)




# def bootstrap_pearson_permutation_elastic(merged_df, cpgs):
#     '''
#     Function that performs bootstrapping and permutation testing using ElasticNet model.
#     Parameters:
#         merged_df (pd.DataFrame): The dataframe containing the features and target.
#         cpgs (list): List of feature column names.
#     Returns:
#         average_bootstrap (float): The average R score from bootstrapping on original data.
#         average_permuted_accuracy (float): The average R score from bootstrapping on permuted data.
#     '''
#     n_testings = 100

#     # Extract features and target
#     X = merged_df[cpgs]
#     y = merged_df['Age']

#     # Arrays to store accuracy values
#     accuracy_values = np.zeros(n_testings)
#     permuted_accuracy_values = np.zeros(n_testings)

#     # Bootstrapping with original data
#     for i in range(n_testings):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
#         model = ElasticNetCV(max_iter=50000, random_state=i)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         pearson_coef, _ = scipy.stats.pearsonr(y_test, y_pred)
#         accuracy_values[i] = pearson_coef

#     average_bootstrap = np.mean(accuracy_values)

#     # Bootstrapping with permuted data
#     valid_permuted_counts = 0  # To keep track of valid iterations
#     for i in range(n_testings):
#         y_permuted = shuffle(y, random_state=i)
#         X_train_permuted, X_test_permuted, y_train_permuted, y_test_permuted = train_test_split(X, y_permuted, test_size=0.2, random_state=i)
#         model = ElasticNetCV(max_iter=50000, random_state=i)
#         model.fit(X_train_permuted, y_train_permuted)
#         y_pred_permuted = model.predict(X_test_permuted)

#         # Check for constant arrays before calculating Pearson correlation
#         if np.all(y_test_permuted == y_test_permuted[0]) or np.all(y_pred_permuted == y_pred_permuted[0]):
#             print(f"Skipping iteration {i} due to constant array.")
#             continue

#         pearson_coef_permuted, _ = scipy.stats.pearsonr(y_test_permuted, y_pred_permuted)
#         permuted_accuracy_values[valid_permuted_counts] = pearson_coef_permuted
#         valid_permuted_counts += 1

#     average_permuted_accuracy = np.mean(permuted_accuracy_values[:valid_permuted_counts])

#     # Plotting
#     plt.figure(figsize=(14, 10))
#     sns.kdeplot(accuracy_values, label='Original data (non-permuted)', fill=True, common_norm=False, color='blue')
#     sns.kdeplot(permuted_accuracy_values[:valid_permuted_counts], color='red', label='Permuted Data', fill=True, common_norm=False)

#     plt.axvline(average_bootstrap, color='blue', linestyle='dashed', linewidth=2, label='Avg r score non permuted')
#     plt.axvline(average_permuted_accuracy, color='red', linestyle='dashed', linewidth=2, label='Avg r score permuted')

#     plt.legend()
#     plt.text(average_bootstrap, plt.ylim()[1] * 0.5, f'{average_bootstrap:.2f}', color='blue', fontsize=12, ha='right')
#     plt.text(average_permuted_accuracy, plt.ylim()[1] * 0.5, f'{average_permuted_accuracy:.2f}', color='red', fontsize=12, ha='right')
#     plt.xlabel('R Score')
#     plt.title('Comparison of Original and Permuted Data R Scores')
#     plt.show()

#     return average_bootstrap, average_permuted_accuracy

# # Example usage
# # merged_df = pd.read_csv('your_data.csv')
# # cpgs = ['list', 'of', 'feature', 'columns']

# # Perform cross-validation evaluation
# cross_validation_evaluation(merged_df, cpgs)

# # Perform bootstrap evaluation
# bootstrap_evaluation(merged_df, cpgs)
