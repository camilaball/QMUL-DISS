import numpy as np
import pandas as pd
import random
import math
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from boruta import BorutaPy
import pandas as pd
import os
import pickle
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
import math 
import random

#CV
from sklearn.utils import safe_sqr
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Feature Select
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV, RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


#Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import LassoCV, Lasso

from sklearn.metrics import r2_score
from warnings import filterwarnings
filterwarnings('ignore')

from collections import Counter

# Ensure compatibility with older versions of NumPy
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

def creation(data):
    total_n_feat = len(data.columns)
    n_feats_per_creat = 50
    n_creatures = 3000
    initial_dict = {}
    for i in range(n_creatures):
        creat_feat = random.sample(list(data.columns), n_feats_per_creat)
        initial_dict[i] = [-999, creat_feat]

    initial_population = pd.DataFrame(initial_dict).transpose()
    initial_population = initial_population.rename(columns={0: "Score", 1: "Feat"})
    
    return initial_population

def training_creatures(X_train, y_train, X_test, y_test, initial_population):
    for i, r in initial_population.iterrows():
        if initial_population.at[i, 'Score'] > 0 or math.isnan(initial_population.at[i, 'Score']):
            continue
        else:
            cur_feat = r[1]
            model = ElasticNet().fit(X_train[cur_feat], y_train)
            y_pred = model.predict(X_test[cur_feat])
            acc = (np.corrcoef(y_test, y_pred)[1][0])**2
            if math.isnan(acc):
                acc = 0.000000001
            initial_population.at[i, 'Score'] = acc
    return initial_population

def cull_mate_mutate(data, cur_pop):
    to_breed = len(cur_pop) - len(cur_pop.dropna())
    cur_pop = cur_pop.dropna()
    n_cull = int(0.50 * len(cur_pop))
    to_breed += n_cull
    
    cur_pop = cur_pop.sort_values(by=['Score'], ascending=False).iloc[:-n_cull, :].reset_index(drop=True)
    fitness_list = list(cur_pop['Score'])
    viable_parents = list(cur_pop.index)
    
    for i in range(to_breed):
        cur_parents = random.choices(viable_parents, weights=np.array(fitness_list) / sum(fitness_list), k=2)
        while cur_parents[0] == cur_parents[1]:
            cur_parents = random.choices(viable_parents, weights=np.array(fitness_list) / sum(fitness_list), k=2)
        parent1_genes = cur_pop.iloc[cur_parents[0]][1]
        parent2_genes = cur_pop.iloc[cur_parents[1]][1]
        set_parents_genes = list(set(list(parent1_genes) + list(parent2_genes)))
        child_genes = list(random.sample(set_parents_genes, len(parent1_genes)))

        if random.random() < 0.2:
            non_mutant_genes = list(random.sample(child_genes, int(0.7 * len(child_genes))))
            possible_mutants = set(data.columns) - set(non_mutant_genes)
            mutant_genes = list(random.sample(possible_mutants, len(parent1_genes) - len(non_mutant_genes)))
            child_genes = mutant_genes + non_mutant_genes
        
        child = pd.DataFrame({'Score': [-999], 'Feat': [child_genes]})
        cur_pop = cur_pop.append(child, ignore_index=True)
    
    return cur_pop

def ga(X_train, y_train, X_test, y_test):
    cur_creatures = creation(X_train)
    fitness = max(cur_creatures['Score'])
    generation = 0
    while generation < 100:
        cur_trained = training_creatures(X_train, y_train, X_test, y_test, cur_creatures)
        cur_creatures = cull_mate_mutate(X_train, cur_trained)
        fitness = max(cur_creatures['Score'])
        generation += 1
        print(f'Generation {generation}: {fitness}')

    best_acc = 0
    best_cpgs = []
    for i in range(10):
        elas = ElasticNet()
        param_grid = {
            "max_iter": [100, 500, 1000],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": np.arange(0.0, 1.0, 0.1)
        }
        grid = GridSearchCV(estimator=elas, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1)
        grid.fit(X_train[cur_creatures['Feat'][i]].to_numpy(), y_train)
        best_parameters = grid.best_params_

        final_clock_model = ElasticNet(
            alpha=best_parameters['alpha'], 
            l1_ratio=best_parameters['l1_ratio'], 
            max_iter=best_parameters['max_iter']
        )
        final_clock_model.fit(X_train[cur_creatures['Feat'][i]].to_numpy(), y_train)
        y_pred = final_clock_model.predict(X_test[cur_creatures['Feat'][i]].to_numpy())
        acc = (np.corrcoef(y_test, y_pred)[1][0])**2
        if acc > best_acc:
            best_cpgs = cur_creatures['Feat'][i]

    return best_cpgs

def feature_selection(filepath):
    input_data = filepath
    X = input_data.drop(['Age'], axis=1)
    y = input_data['Age']

    ga_list = []
    inter = []

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    counter = 1
    for train_indices, test_indices in cv.split(X):
        score_dict = {}
        X_train = X.iloc[train_indices, :]
        X_test = X.iloc[test_indices, :]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        ga_cpgs = ga(X_train, y_train, X_test, y_test)
        score_dict['GA de novo'] = [ga_cpgs]
        ga_list += list(ga_cpgs)
#         if counter == 1:
#             ga_intersect = ga_cpgs
#         else:
#             ga_intersect = list(set(ga_intersect) & set(ga_cpgs))

#         inter_name, inter_selected_cpgs = training_intersected_cpgs(X_train, y_train, score_dict)
#         score_dict[inter_name] = [inter_selected_cpgs]
#         inter += list(inter_selected_cpgs)
#         if counter == 1:
#             inter_intersect = inter_selected_cpgs
#         else:
#             inter_intersect = list(set(inter_intersect) & set(inter_selected_cpgs))

#         counter += 1

    d = {'ga': list(ga_list)}

    final_cpgs_df = pd.DataFrame()
    name_list = []
    for method in d.keys():
        cur_cpg_list = pd.Series(d[method])
        name_list.append(method)
        final_cpgs_df = pd.concat([final_cpgs_df, cur_cpg_list], ignore_index=True, axis=1)
    final_cpgs_df.to_csv('results_20_cv2_open_wgbs.csv', index=False)
    print('Finished',ga_list)

