# Code from the github page of the publication "Novel feature selection methods for construction of accurate epigenetic clocks"

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

np.int = np.int32
np.float = np.float64
np.bool = np.bool_
def boruta(X, y):
    name = 'Boruta de novo'
    rf = RandomForestRegressor()
    if len(X.columns) > 1500:
        selector = BorutaPy(estimator = rf, n_estimators = 4, verbose=10).fit(np.array(X), np.array(y))
        selected_cpgs = list(X.columns[selector.support_])   
        return(name, selected_cpgs)
    
    elif len(X.columns) <= 1500:
        selector = BorutaPy(estimator = rf, n_estimators = 'auto', verbose=10).fit(np.array(X), np.array(y))
        selected_cpgs = list(X.columns[selector.support_])   
        return(name, selected_cpgs)

def preselected_with_boruta(X, y, selected_cpgs, name):
    origin = name
    X = X[selected_cpgs]
    if len(X.columns) <= 0:
        return(origin, [])
    else:
        name, new_cpgs = boruta(X,y)
        return(origin, new_cpgs)

def SFMElastic(X, y, X_test, y_test):
    name = "SFM Elastic de novo"
    thresh_list = [0.01, 0.05, 0.1, 0.5]
    elas = ElasticNet()
    best_score= 0
    best_cpgs = []
    
    for i in thresh_list:
        print("Completing SFM with threshold: " +str(i))
        selector = SelectFromModel(elas, threshold=i).fit(X, y)
        feature_idx = selector.get_support(indices=True)
        selected_cpgs = X.columns[feature_idx]
        
        if len(selected_cpgs) == 0:
            continue
        else:
            elas = ElasticNet()
            elas.fit(X[selected_cpgs], y)
            y_pred = elas.predict(X_test[selected_cpgs])
            acc = (np.corrcoef(y_test, y_pred)[1][0])**2
            if acc > best_score:
                best_cpgs = selected_cpgs
    return(name, best_cpgs)

def SFMExtra(X, y, X_test, y_test):
    name = "SFM ExtraTrees de novo"
    thresh_list = [0.01, 0.05, 0.1, 0.5]
    clf = ExtraTreesRegressor(n_estimators=8)
    best_score= 0
    best_cpgs = []
    
    for i in thresh_list:
        print("Completing SFM with threshold: " +str(i))
        selector = SelectFromModel(clf, threshold=i).fit(X, y)
        feature_idx = selector.get_support(indices=True)
        selected_cpgs = X.columns[feature_idx]
        
        if len(selected_cpgs) == 0:
            continue
        else:
            elas = ElasticNet()
            elas.fit(X[selected_cpgs], y)
            y_pred = elas.predict(X_test[selected_cpgs])
            acc = (np.corrcoef(y_test, y_pred)[1][0])**2
            if acc > best_score:
                best_cpgs = selected_cpgs
    return(name, best_cpgs)
        

def RFE100(X, y, n_features_to_select = 100):
    name = 'RFE de novo to 100'
    elas = ElasticNet()
    estimator = elas
    n_features = X.shape[1]
    n_features_to_select = n_features_to_select
    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)
    step = 0.01

    while np.sum(support_) > n_features_to_select:
        step = 0.01
        features = np.arange(n_features)[support_]
        estimator = clone(estimator)
        print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X.iloc[:,features], y)   
        step = int(max(1, step * np.sum(support_)))
        print("Eliminating "+str(step)+ " features")

        importances = estimator.coef_
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)

        ranks = np.argsort(importances)
        ranks = np.ravel(ranks)
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1

    features = np.arange(n_features)[support_]
    estimator_ = clone(estimator)
    final_model = estimator_.fit(X.iloc[:,features], y)

    end_support = support_
    end_ranking = ranking_
    n_features_ = support_.sum()
    feature_name = X.columns[end_support]
    
    selected_cpgs = list(feature_name)
        
    return(name, selected_cpgs)  

def RFE1500(X, y, n_features_to_select = 1500):
    name = 'RFE de novo to 1500'
    elas = ElasticNet()
    estimator = elas
    n_features = X.shape[1]
    n_features_to_select = n_features_to_select
    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)
    step = 0.01

    while np.sum(support_) > n_features_to_select:
        step = 0.01
        features = np.arange(n_features)[support_]
        estimator = clone(estimator)
        print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X.iloc[:,features], y)   
        step = int(max(1, step * np.sum(support_)))
        print("Eliminating "+str(step)+ " features")

        importances = estimator.coef_
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)

        ranks = np.argsort(importances)
        ranks = np.ravel(ranks)
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1


    features = np.arange(n_features)[support_]
    estimator_ = clone(estimator)
    final_model = estimator_.fit(X.iloc[:,features], y)

    end_support = support_
    end_ranking = ranking_
    n_features_ = support_.sum()
    feature_name = X.columns[end_support]
    
    selected_cpgs = list(feature_name)
    return(selected_cpgs)  
    
def training_intersected_cpgs(X,y,score_dict):
    name = 'Intersection of all selected CpGs'
    all_cpgs = []
    for name in score_dict.keys():
        all_cpgs += list(score_dict[name][0])
    all_cpgs = list(pd.Series(all_cpgs).dropna())
    intersected_cpgs = list(set(all_cpgs))
    return(name, intersected_cpgs)

def RFE10k(X, y, n_features_to_select = 10000):
    name = 'RFE de novo to 10000'
    elas = ElasticNet()
    estimator = elas
    n_features = X.shape[1]
    n_features_to_select = n_features_to_select
    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)
    step = 0.01

    while np.sum(support_) > n_features_to_select:
        step = 0.01
        features = np.arange(n_features)[support_]
        estimator = clone(estimator)
        print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X.iloc[:,features], y)   
        step = int(max(1, step * np.sum(support_)))
        print("Eliminating "+str(step)+ " features")

        importances = estimator.coef_
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)

        ranks = np.argsort(importances)
        ranks = np.ravel(ranks)
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1


    features = np.arange(n_features)[support_]
    estimator_ = clone(estimator)
    final_model = estimator_.fit(X.iloc[:,features], y)

    end_support = support_
    end_ranking = ranking_
    n_features_ = support_.sum()
    feature_name = X.columns[end_support]
    
    selected_cpgs = list(feature_name)
    return(selected_cpgs) 

def creation(data): #Make no. of creatures, no. features etc. arguements
    total_n_feat = len(data.columns)
    n_feats_per_creat = 50
    n_creatures = 3000 #int(n_feats_per_creat**3)
    initial_dict = {}
    for i in range(n_creatures):
        creat_feat = random.sample(list(data.columns), n_feats_per_creat)
        initial_dict[i] = [-999, creat_feat]

    initial_population = pd.DataFrame(initial_dict)
    initial_population = initial_population.transpose()
    initial_population = initial_population.rename(columns={0: "Score", 1: "Feat"})
    
    return initial_population

def training_creatures(X_train, y_train, X_test, y_test, initial_population):
    
    for i, r in initial_population.iterrows():
        if initial_population.at[i, 'Score'] > 0 or math.isnan(initial_population.at[i, 'Score']):
            continue
        else:
            cur_feat = r[1]
            model = ElasticNet().fit(X_train[cur_feat],y_train)
            y_pred = model.predict(X_test[cur_feat])
            
            acc = (np.corrcoef(y_test, y_pred)[1][0])**2
            if math.isnan(acc) == True:
                acc = 0.000000001
                initial_population.at[i, 'Score'] = acc
            else:
                initial_population.at[i, 'Score'] = acc
    return initial_population

def cull_mate_mutate(data, cur_pop):
    to_breed = len(cur_pop) - len(cur_pop.dropna())
    cur_pop = cur_pop.dropna()
    n_cull = int(0.50*len(cur_pop))
    to_breed += n_cull
    
    cur_pop = cur_pop.sort_values(by=['Score'], ascending = False).iloc[:-n_cull, :]
    cur_pop = cur_pop.reset_index().drop(['index'], axis=1)
    fitness_list = list(cur_pop['Score'])
    viable_parents = list(cur_pop.index)
    
    for i in range(to_breed):
        cur_parents = random.choices(viable_parents, weights=np.array(fitness_list)/sum(fitness_list), k= 2)
        while cur_parents[0] == cur_parents[1]:
            cur_parents = random.choices(viable_parents, weights=np.array(fitness_list)/sum(fitness_list), k= 2)
        parent1_genes = cur_pop.iloc[cur_parents[0]][1]
        parent2_genes = cur_pop.iloc[cur_parents[1]][1]
        set_parents_genes = list(set(list(parent1_genes) + list(parent2_genes)))
        child_genes = list(random.sample(set_parents_genes, len(parent1_genes)))

        if random.randrange(0,100,1)/100 < 0.2:
            non_mutant_genes = list(random.sample(child_genes, int(0.7*len(child_genes))))
            possible_mutants = set(data.columns) - set(non_mutant_genes)
            mutant_genes = list(random.sample(possible_mutants, len(parent1_genes) - len(non_mutant_genes)))
            child_genes = mutant_genes + non_mutant_genes
            
            child = {'Score': -999, 'Feat': child_genes}
            child = pd.DataFrame({'Score': [-999], 'Feat': [child_genes]})
            cur_pop = pd.concat([cur_pop, child], ignore_index=True)
            
        else:
            child = {'Score': -999, 'Feat': child_genes}
            child = pd.DataFrame({'Score': [-999], 'Feat': [child_genes]})
            cur_pop = pd.concat([cur_pop, child], ignore_index=True)
        
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
        print('Generation ' + str(generation), fitness)

    best_acc = 0
    best_cpgs = []
    for i in range(10):

        elas = ElasticNet()
        param_grid = {"max_iter": [100, 500, 1000],
                  "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": np.arange(0.0, 1.0, 0.1)}

        grid = GridSearchCV(estimator=elas,
                             param_grid=param_grid,
                             scoring='r2',
                             cv=10,
                             n_jobs=-1)
        grid.fit(X_train[cur_creatures['Feat'][i]].to_numpy(), y_train)
        best_parameters = grid.best_params_

        final_clock_model = ElasticNet(alpha = best_parameters['alpha'], l1_ratio = best_parameters['l1_ratio'], max_iter = best_parameters['max_iter'])
        final_clock_model.fit(X_train[cur_creatures['Feat'][i]].to_numpy(), y_train)
        y_pred = final_clock_model.predict(X_test[cur_creatures['Feat'][i]].to_numpy())

        acc = (np.corrcoef(y_test, y_pred)[1][0])**2
        if acc > best_acc:
            best_cpgs = cur_creatures['Feat'][i]

    return best_cpgs

def feature_selection(filepath,percentage,foldss,clock):
    input_data = pd.read_csv(filepath)
    X = input_data.drop(['Age'], axis = 1)
    y = input_data['Age']

    boruta_list = []
    rfe1500_to_boruta = []
    rfe100 = []
    sfm_elas = []
    sfm_elas_boruta = []
    sfm_extra = []
    sfm_extra_boruta = []
    rfe1500_to_sfm = []
    rfe1000_to_rfecv = []
    Kbest_list_25 = []
    Kbest_list_2000_boruta = []
    basic_elas_list = []
    rfe10k_ga_list = []
    ga_list = []
    inter = []
    inter_boruta =[]

    cv = KFold(n_splits = foldss, shuffle = True, random_state = 0)
    counter = 1
    for train_indices, test_indices in cv.split(X):
        score_dict = {}
        X_train = X.iloc[train_indices, :]
        X_test = X.iloc[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        boruta_name, boruta_selected_cpgs = boruta(X_train,y_train)
        score_dict[boruta_name] = [boruta_selected_cpgs]
        boruta_list += list(boruta_selected_cpgs)
        if counter == 1:
            boruta_intersect = boruta_selected_cpgs
        else: 
            boruta_intersect = list(set(boruta_intersect) & set(boruta_selected_cpgs))
        #_____________________________________________________________________________________________________________________________________

        rfe_1500_cpgs = RFE1500(X_train, y_train)
        rfe_1000_cpgs = RFE1500(X_train[rfe_1500_cpgs],y_train, 1000) #Use the 1500 before to go to 1000

        elas = ElasticNet()
        selector = RFECV(estimator = elas, step=1, cv = 10, scoring = 'r2').fit(X_train[rfe_1000_cpgs], y_train)
        feature_idx = selector.get_support(indices=True)
        rfe_1000_to_rfecv_cpgs = list(X_train[rfe_1000_cpgs].columns[feature_idx])
        score_dict['RFE de novo to 1000 followed by RFECV'] = [rfe_1000_to_rfecv_cpgs]
        rfe1000_to_rfecv += list(rfe_1000_to_rfecv_cpgs)
        if counter == 1:
            rfe1000_to_rfecv_intersect = rfe_1000_to_rfecv_cpgs
        else: 
            rfe1000_to_rfecv_intersect = list(set(rfe1000_to_rfecv_intersect) & set(rfe_1000_to_rfecv_cpgs))

        rfe_w_boruta_name, rfe_w_boruta_selected_cpgs = preselected_with_boruta(X_train, y_train, rfe_1500_cpgs, 'RFE de novo to 1500 followed by Boruta')
        score_dict[rfe_w_boruta_name] = [rfe_w_boruta_selected_cpgs]
        rfe1500_to_boruta += list(rfe_w_boruta_selected_cpgs)
        if counter == 1:
            rfe1500_to_boruta_intersect = rfe_w_boruta_selected_cpgs
        else: 
            rfe1500_to_boruta_intersect = list(set(rfe1500_to_boruta_intersect) & set(rfe_w_boruta_selected_cpgs))

        rfe_name, rfe_selected_cpgs = RFE100(X_train[rfe_1000_cpgs],y_train)
        score_dict[rfe_name] = [rfe_selected_cpgs]
        rfe100 += list(rfe_selected_cpgs)
        if counter == 1:
            rfe100_intersect = rfe_selected_cpgs
        else: 
            rfe100_intersect = list(set(rfe100_intersect) & set(rfe_selected_cpgs))
        #_____________________________________________________________________________________________________________________________________

        sfm_elas_name, sfm_elas_selected_cpgs = SFMElastic(X_train, y_train, X_test, y_test)
        score_dict[sfm_elas_name] = [sfm_elas_selected_cpgs]
        sfm_elas += list(sfm_elas_selected_cpgs)
        if counter == 1:
            sfm_elas_intersect = sfm_elas_selected_cpgs
        else: 
            sfm_elas_intersect = list(set(sfm_elas_intersect) & set(sfm_elas_selected_cpgs))

        sfm_elas_w_boruta_name, sfm_elas_w_boruta_selected_cpgs = preselected_with_boruta(X_train, y_train, sfm_elas_selected_cpgs, 'SFMElastic de novo followed by Boruta')
        score_dict[sfm_elas_w_boruta_name] = [sfm_elas_w_boruta_selected_cpgs]
        sfm_elas_boruta += list(sfm_elas_w_boruta_selected_cpgs)

        if counter == 1:
            sfm_elas_boruta_intersect = sfm_elas_w_boruta_selected_cpgs
        else: 
            sfm_elas_boruta_intersect = list(set(sfm_elas_boruta_intersect) & set(sfm_elas_w_boruta_selected_cpgs))
        #_____

        sfm_extra_name, sfm_extra_selected_cpgs = SFMExtra(X_train, y_train, X_test, y_test)
        score_dict[sfm_extra_name] = [sfm_extra_selected_cpgs]
        sfm_extra += list(sfm_extra_selected_cpgs)
        if counter == 1:
            sfm_extra_intersect = sfm_extra_selected_cpgs
        else: 
            sfm_extra_intersect = list(set(sfm_extra_intersect) & set(sfm_extra_selected_cpgs))

        sfm_extra_w_boruta_name, sfm_extra_w_boruta_selected_cpgs = preselected_with_boruta(X_train, y_train, sfm_extra_selected_cpgs, 'SFMExtra de novo followed by Boruta')
        score_dict[sfm_extra_w_boruta_name] = [sfm_extra_w_boruta_selected_cpgs]
        sfm_extra_boruta += list(sfm_extra_w_boruta_selected_cpgs)
        if counter == 1:
            sfm_extra_boruta_intersect = sfm_extra_w_boruta_selected_cpgs
        else: 
            sfm_extra_boruta_intersect = list(set(sfm_extra_boruta_intersect) & set(sfm_extra_w_boruta_selected_cpgs))

        rfe_1500_to_sfm_name, rfe_1500_to_sfm_selected_cpgs = SFMElastic(X_train[rfe_1500_cpgs], y_train, X_test, y_test)
        score_dict['RFE de novo to 1500 followed by SFM'] = [rfe_1500_to_sfm_selected_cpgs] #might need to turn into a list
        rfe1500_to_sfm += list(rfe_1500_to_sfm_selected_cpgs)
        if counter == 1:
            rfe1500_to_sfm_intersect = rfe_1500_to_sfm_selected_cpgs
        else: 
            rfe1500_to_sfm_intersect = list(set(rfe1500_to_sfm_intersect) & set(rfe_1500_to_sfm_selected_cpgs))
        #_____________________________________________________________________________________________________________________________________

        selector = SelectKBest(score_func=f_regression, k=25).fit(X_train, y_train)
        feature_idx = selector.get_support(indices=True)
        selected_cpgs_25 = X_train.columns[feature_idx]
        score_dict['KBest 25'] = [selected_cpgs_25]
        Kbest_list_25 += list(selected_cpgs_25)
        if counter == 1:
            Kbest_list_25_intersect = selected_cpgs_25
        else: 
            Kbest_list_25_intersect = list(set(Kbest_list_25_intersect) & set(selected_cpgs_25))


        selector = SelectKBest(score_func=f_regression, k=2000).fit(X_train, y_train)
        feature_idx = selector.get_support(indices=True)
        selected_cpgs_2000 = X_train.columns[feature_idx]
        Kbest_name, Kbest_boruta_cpgs = preselected_with_boruta(X_train, y_train, selected_cpgs_2000, 'KBest de novo followed by Boruta')
        score_dict['KBest 2000 then Boruta'] = [Kbest_boruta_cpgs]
        Kbest_list_2000_boruta += list(Kbest_boruta_cpgs)
        if counter == 1:
            Kbest_list_2000_boruta_intersect = Kbest_boruta_cpgs
        else: 
            Kbest_list_2000_boruta_intersect = list(set(Kbest_list_2000_boruta_intersect) & set(Kbest_boruta_cpgs))

        basic_elas = ElasticNet().fit(X_train, y_train)
        coef = pd.DataFrame(basic_elas.coef_, index = X_train.columns)
        coef = coef.reset_index()
        basic_elas_cpgs = list(coef[coef[0] != 0].iloc[:,0])
        score_dict['Basic ElasticNet'] = [basic_elas_cpgs]
        basic_elas_list += list(basic_elas_cpgs)
        if counter == 1:
            basic_elas_intersect = basic_elas_cpgs
        else: 
            basic_elas_intersect = list(set(basic_elas_intersect) & set(basic_elas_cpgs))


        rfe_10k_cpgs = RFE10k(X_train, y_train)
        rfe10k_ga_cpgs = ga(X_train[rfe_10k_cpgs], y_train, X_test[rfe_10k_cpgs], y_test)
        score_dict['RFE10K then GA'] = [rfe10k_ga_cpgs]
        rfe10k_ga_list += list(rfe10k_ga_cpgs)
        if counter == 1:
            rfe10k_ga_intersect = rfe10k_ga_cpgs
        else: 
            rfe10k_ga_intersect = list(set(rfe10k_ga_intersect) & set(rfe10k_ga_cpgs))


        ga_cpgs = ga(X_train, y_train, X_test, y_test)
        score_dict['GA de novo'] = [ga_cpgs]
        ga_list += list(ga_cpgs)
        if counter == 1:
            ga_intersect = ga_cpgs
        else: 
            ga_intersect = list(set(ga_intersect) & set(ga_cpgs))

        #_____________________________________________________________________________________________________________________________________

        inter_name, inter_selected_cpgs = training_intersected_cpgs(X_train, y_train, score_dict)    
        score_dict[inter_name] = [inter_selected_cpgs]
        inter += list(inter_selected_cpgs)
        if counter == 1:
            inter_intersect = inter_selected_cpgs
        else: 
            inter_intersect = list(set(inter_intersect) & set(inter_selected_cpgs))

        inter_w_boruta_name, inter_w_boruta_selected_cpgs = preselected_with_boruta(X_train, y_train, inter_selected_cpgs, 'Intersection followed by Boruta')
        score_dict[inter_w_boruta_name] = [inter_w_boruta_selected_cpgs]
        inter_boruta += list(inter_w_boruta_selected_cpgs)
        if counter == 1:
            inter_boruta_intersect = inter_w_boruta_selected_cpgs
        else: 
            inter_boruta_intersect = list(set(inter_boruta_intersect) & set(inter_w_boruta_selected_cpgs))
        #_____________________________________________________________________________________________________________________________________

        counter+=1


    d = {'boruta': list(boruta_list), 
         'rfe1500_to_boruta': list(rfe1500_to_boruta),
         'rfe_1000_to_rfecv_cpgs': list(rfe1000_to_rfecv),
         'rfe100' : list(rfe100),
         'sfm_elas': list(sfm_elas),
         'sfm_elas_boruta': list(sfm_elas_boruta),
         'sfm_extra': list(sfm_extra),
         'sfm_extra_boruta': list(sfm_extra_boruta),
         'rfe1500_to_sfm': list(rfe1500_to_sfm),
         'kbest25': list(Kbest_list_25),
         'kbest2k_boruta': list(Kbest_list_2000_boruta),
         'basic_elas': list(basic_elas_list),
         'ga': list(ga_list),
         'rfe10k_ga_list' : list(rfe10k_ga_list),
         'inter': list(inter),
         'inter_boruta': list(inter_boruta)}

    final_cpgs_df = pd.DataFrame()
    name_list = []
    for method in d.keys():
        cur_cpg_list = pd.Series(d[method])
        name_list.append(method)
        final_cpgs_df = pd.concat([final_cpgs_df, cur_cpg_list], ignore_index=True, axis=1)
    final_cpgs_df.to_csv(f'~/research_project/src/imputation_tests_open_rrbs/feature_selection/results_all_feature_selections_{percentage}__{foldss}CV.csv',index=False)
    print('Finished')
    
# feature_selection('~/research_project/src/results/merged_20_epi_rrbs_only.csv')
