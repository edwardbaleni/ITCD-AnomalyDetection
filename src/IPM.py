# %%

# TODO: We have the hyperparameter setting.
# Now we need to use those to get the IPM measures.

# %%



# Select

# -*- coding: utf-8 -*-

import numpy as np
import sys

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import average_precision_score, roc_auc_score
from time import time
# from sklearn.datasets import one_class_data

from scipy.io import loadmat
from scipy.stats import rankdata
import os
from scipy.stats import kendalltau

mat_file_list = [
    'annthyroid.mat',
    'arrhythmia.mat',
    'breastw.mat',
    'glass.mat',
    'ionosphere.mat',
    'letter.mat',
    'lympho.mat',
    'mammography.mat',
    'mnist.mat',
    'musk.mat',
    'optdigits.mat',
    'pendigits.mat',
    'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    'speech.mat',
    'thyroid.mat',
    'vertebral.mat',
    'vowels.mat',
    'wbc.mat',
    'wine.mat',
]

arff_file_list = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    # 'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',
    'ALOI', # too large
    'Glass', # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC', # too small
    'WDBC', # too small
    'WPBC', # too small
]

moving_size = 3
perf_mat = np.zeros([len(mat_file_list), 3])

time_tracker = []
for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    # loading and vectorization
    mat = loadmat(os.path.join("data", "ODDS", mat_file))
    score_mat = np.loadtxt(os.path.join("scores_mat", mat_file+'.csv'), delimiter=',')
    
    t0 = time()
    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat

    X = mat['X']
    y = mat['y'].ravel()
    
    n_samples, n_models = score_mat.shape[0], score_mat.shape[1]
    
    # build target vector 
    target = np.mean(inv_rank_mat, axis=1)
    
    kendall_vec = np.full([n_models,], -99).astype(float)
    kendall_tracker = []
    
    model_ind = list(range(n_models))
    selected_ind = []
    last_kendall = 0
    
    # build the first target
    for i in model_ind:
        kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
    
    most_sim_model = np.argmax(kendall_vec)
    kendall_tracker.append(np.max(kendall_vec))
    
    # option 1: last one: keep increasing/non-decreasing
    # last_kendall = kendall_tracker[-1]
    
    # # option 2: moving avg
    # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
    
    # option 3: average of all
    last_kendall = np.mean(kendall_tracker)
    
    selected_ind.append(most_sim_model)
    model_ind.remove(most_sim_model)
    
    
    while len(model_ind) != 0:
    
        target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
        kendall_vec = np.full([n_models,], -99).astype(float)
        
        for i in model_ind:
            kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
            
        most_sim_model = np.argmax(kendall_vec)
        max_kendall = np.max(kendall_vec)
        
        if max_kendall >= last_kendall:
            selected_ind.append(most_sim_model)
            model_ind.remove(most_sim_model)
            kendall_tracker.append(max_kendall)
            
            # option 1: last one: keep increasing/non-decreasing
            # last_kendall = kendall_tracker[-1]
            
            # # option 2: moving avg
            # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
            
            # option 3: average of all
            last_kendall = np.mean(kendall_tracker)

        else:
            break
    
    final_target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
    average_target = np.mean(inv_rank_mat, axis=1)
    
    
    print('SELECT', mat_file, roc_auc_score(y, final_target*-1), 
          average_precision_score(y, final_target*-1), 
          precision_n_scores(y, final_target*-1))
    print('Average', mat_file, roc_auc_score(y, average_target*-1), 
          average_precision_score(y, average_target*-1), 
          precision_n_scores(y, average_target*-1))
    print()
    
    perf_mat[j, 0] = roc_auc_score(y, final_target*-1)
    perf_mat[j, 1] = average_precision_score(y, final_target*-1)
    perf_mat[j, 2] = precision_n_scores(y, final_target*-1)
    
    similarity = []
    for k in range(n_models):
        similarity.append(kendalltau(final_target, inv_rank_mat[:, k])[0])
        
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Model.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Target.csv'), final_target*-1, delimiter=',')


# %%

# MC


# -*- coding: utf-8 -*-


import os 
from time import time
import pandas as pd
import numpy as np
from base_detectors import get_detectors
from copy import deepcopy
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
clf_df = pd.read_csv('roc_mat_2.csv', low_memory=False)
headers = clf_df.columns.tolist()[4:]


mat_file_list = [
    'annthyroid',
    'arrhythmia',
    'breastw',
    'glass',
    'ionosphere',
    'letter',
    'lympho',
    'mammography',
    'mnist',
    'musk',
    'optdigits',
    'pendigits',
    'pima',
    'satellite',
    'satimage-2',
    'speech',
    'thyroid',
    'vertebral',
    'vowels',
    'wbc',
    'wine',
]


arff_file_list = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',
    'ALOI', # too large
    'Glass', # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC', # too small
    'WDBC', # too small
    'WPBC', # too small
]

time_tracker = []
#1111111111111111111111111111111#
for mat_file in mat_file_list:
# for mat_file in arff_file_list:
    print(mat_file)
    #22222222222222222222222222222222#
    # mat =  mat_file + '.csv'
    mat =  mat_file + '.mat.csv'
    mat_X = mat_file + '.mat_X.csv'
    mat_y = mat_file + '.mat_y.csv'

    #333333333333333333333333333333333333333#
    t0 = time()
    # df = pd.read_csv(os.path.join("scores_arff", mat), names=headers)    
    df = pd.read_csv(os.path.join("scores_mat", mat), names=headers,low_memory=False)
    output_mat = df.to_numpy().astype('float64')

    output_mat = np.nan_to_num(output_mat)
    output_mat_r = rankdata(output_mat, axis=0)

    output_mat = MinMaxScaler().fit_transform(output_mat_r)
    base_detectors, randomness_flags =  get_detectors()
    
    base_detectors_ranges = {}
    

    keys = list(range(8))
    base_detectors_ranges[0] = list(range(0, 54))
    base_detectors_ranges[1] = list(range(54, 61))
    base_detectors_ranges[2] = list(range(61, 142))
    base_detectors_ranges[3] = list(range(142, 178))
    base_detectors_ranges[4] = list(range(178, 214))
    base_detectors_ranges[5] = list(range(214, 254))
    base_detectors_ranges[6] = list(range(254, 290))
    base_detectors_ranges[7] = list(range(290, 297))
    
    sum_check = 0
    key_len = []
    
    for i in keys:
        sum_check += len(base_detectors_ranges[i])
        key_len.append(len(base_detectors_ranges[i]))
    assert (sum_check==297)
    
    m = len(base_detectors)
    
    similar_mat = np.full((m, m), 1).astype(float)

    for i in keys:
        # get all the configuration with the same hyperparameters
        same_hypers = base_detectors_ranges[i]
        
        for k in same_hypers:
            temp_list = list(range(m))
            temp_list.remove(k)

            for j in temp_list:
                # corr = ndcg_score([np.nan_to_num(output_mat[:, k])], [np.nan_to_num(output_mat[:, j])])
                # corr = spearmanr(output_mat[:, [k, j]])[0]
                corr = kendalltau(output_mat[:, k], output_mat[:, j])[0]
                similar_mat[k, j] = corr
    
    B = (similar_mat+similar_mat.T)/2
    # fix nan problem
    B = np.nan_to_num(B)
    
    similarity = (np.sum(B, axis=1)-1)/(m-1)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)
    print('kendall')
    # #4444444444444444444444444444444444444444444444#
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')  

    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC1.kendall.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.kendall.csv'), similarity, delimiter=',') 
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'MC.csv'), similarity, delimiter=',')    
    # # y=pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR1.csv')).to_numpy()
    # # w = pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR.csv')).to_numpy()
    # # p = np.concatenate([w,y], axis=1)
    # # spearmanr(p)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# %%

# HITS

# -*- coding: utf-8 -*-

import numpy as np
import sys

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import average_precision_score, roc_auc_score
from time import time
# from sklearn.datasets import one_class_data

from scipy.io import loadmat
from scipy.stats import rankdata
import os
from scipy.stats import kendalltau

mat_file_list = [
    'annthyroid.mat',
    'arrhythmia.mat',
    'breastw.mat',
    'glass.mat',
    'ionosphere.mat',
    'letter.mat',
    'lympho.mat',
    'mammography.mat',
    'mnist.mat',
    'musk.mat',
    'optdigits.mat',
    'pendigits.mat',
    'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    'speech.mat',
    'thyroid.mat',
    'vertebral.mat',
    'vowels.mat',
    'wbc.mat',
    'wine.mat',
]

arff_file_list = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    # 'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',
    'ALOI', # too large
    'Glass', # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC', # too small
    'WDBC', # too small
    'WPBC', # too small
]

# moving_size = 3
perf_mat = np.zeros([len(mat_file_list), 3])

time_tracker = []
for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    # loading and vectorization
    mat = loadmat(os.path.join("data", "ODDS", mat_file))
    score_mat = np.loadtxt(os.path.join("scores_mat", mat_file+'.csv'), delimiter=',')

    t0 = time()
    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat

    X = mat['X']
    y = mat['y'].ravel()
    
    n_samples, n_models = score_mat.shape[0], score_mat.shape[1]

    hub_vec = np.full([n_models, 1],  1/n_models)
    auth_vec = np.zeros([n_samples, 1])
    
    hub_vec_list = []
    auth_vec_list = []
    
    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)
    
    for i in range(500):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)
        
        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)
        
        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]
        
        # print(auth_diff.sum(), auth_diff.mean(), auth_diff.std())
        # print(hub_diff.sum(), hub_diff.mean(), hub_diff.std())
        # print()
        
        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            print('break at', i)
            break
        
        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)
        
    
    print('HITS', mat_file, roc_auc_score(y, auth_vec*-1), 
          average_precision_score(y, auth_vec*-1), 
          precision_n_scores(y, auth_vec*-1))
    print()
    
    perf_mat[j, 0] = roc_auc_score(y, auth_vec*-1)
    perf_mat[j, 1] = average_precision_score(y, auth_vec*-1)
    perf_mat[j, 2] = precision_n_scores(y, auth_vec*-1)

    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Model.csv'), hub_vec, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Target.csv'), auth_vec, delimiter=',')


