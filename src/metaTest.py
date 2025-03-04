# %%
import os
import time
from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd
from combo.models.score_comb import average
from pyod.utils.utility import standardizer
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold

from utils.init_meta import get_meta_init
from utils.utility import get_diff, process_batch, get_sim_kendall

import joblib
from sklearn.preprocessing import RobustScaler




def getTrain(model_name, feature_index, n_hp_configs):
    # XXX: MetaFeatures - Train
    meta = joblib.load(f"results/meta/{model_name}/meta_dataframe.pkl")
    y = joblib.load(f"results/meta/{model_name}/y_meta.pkl")
    cols = meta.columns

    n_datasets = int(meta.shape[0]/n_hp_configs)

    X = np.array(meta)
    X = X.astype(np.float32)
    X = np.nan_to_num(X)

    # load ap rank and ap values
    y = np.array(y)
    y = y.astype(np.float32)
    ap_values = y.reshape(n_datasets, n_hp_configs)
    sorted_ap_val = np.sort(ap_values, axis=1)
    ap_all = ap_values

    # LOOCV; build the train test index. One dataset for test
    kf = KFold(n_splits=(int(n_datasets)))

    train_indexs = []
    test_indexs = []

    inds = []
    all_index = np.arange(X.shape[0])

    counter = 0
    for train_index, test_index in kf.split(X):
        inds.append(counter)
        train_indexs.append(train_index)
        test_indexs.append(test_index)
        counter += 1

    # Build meta-surrogate functions
    gp_clf = GaussianProcessRegressor()
    pre_clfs = []
    for ind, train_index, test_index in zip(inds, train_indexs, test_indexs):
        print('Build meta-surrogate function', ind)

        _, X_test = X[train_index, :], X[test_index, :]
        _, y_test = y[train_index], y[test_index]

        X_test = standardizer(X_test)

        # this creates the meta-surrogate function
        # for the training portion t_i(\lambda_j) |-> Perf_i,j
    
        if isinstance(feature_index,int):
            pre_clfs.append(deepcopy(gp_clf).fit(X_test[:, feature_index].reshape(-1, 1), y_test))
        else:
            pre_clfs.append(deepcopy(gp_clf).fit(X_test[:, feature_index], y_test))



    return cols, n_datasets, pre_clfs, ap_values, sorted_ap_val, ap_all

model_name = 'LOF'

feature_index = None
n_hp_configs = None

    # specify the input features, i.e., HPs, for the surrogate function
if model_name == 'LOF':
    feature_index = list(range(200, 205))
    n_hp_configs = 116
elif model_name == 'IF':
    feature_index = list(range(200, 202))
    n_hp_configs = 228
elif model_name == 'PCA':
    feature_index = 200
    n_hp_configs = 11
elif model_name == 'ABOD':
    feature_index = 200
    n_hp_configs = 19

# set up the random seed
random_seed = 42
random_state = np.random.RandomState(random_seed)

# set the HPs of HPOD
n_epochs = None
if model_name == 'IF' or model_name == 'LOF':
    n_epochs = 50
elif model_name == 'PCA':
    n_epochs = 4
else:
    n_epochs = 10

n_init_configs = 10
n_neighbors = 1

col_names, n_datasets, pre_clfs, ap_values, sorted_ap_val, ap_all = getTrain(model_name, feature_index, n_hp_configs)

# Get test data and set
testMeta = joblib.load(f"results/meta/Test/{model_name}_meta_dataframe.pkl")
del testMeta["Dataset"]
scaler = RobustScaler()
X = np.array(testMeta)
X = X.astype(np.float32)
X = np.nan_to_num(X)

n_datasets = n_tests = 31
kf = KFold(n_splits=(int(n_tests)))

train_indexs = []
test_indexs = []

inds = []

counter = 0
for train_index, test_index in kf.split(X):
    inds.append(counter)
    train_indexs.append(train_index)
    test_indexs.append(test_index)
    counter += 1

# track HPOD's performance
ours = []

best_model_history = np.zeros([n_tests, n_epochs]).astype('float')
best_model_index = np.zeros([n_tests, n_epochs]).astype('int')
ei_tracker = np.zeros([n_tests, n_epochs]).astype('float')

# this gives us the indices of the initial models
init_configs_list, _ = get_meta_init(n_init_models=n_init_configs, ap_values=ap_values)

OOS_params = pd.DataFrame()

for ind, train_index, test_index in zip(inds, train_indexs, test_indexs):
    # meta init
    init_configs = init_configs_list[ind]
    # remove duplicates
    init_configs = list(set(init_configs))

    # initialize the evaluation set
    all_models = list(range(ap_values.shape[1]))
    curr_models = init_configs
    # cur_models are the models that are now being considered
    # left models would be the models that are not yet considered
    left_models = get_diff(all_models, curr_models)

    X_test = X[test_index, :]
    # X_standardized = standardizer(X[test_index, :])
    X_test_standardized = standardizer(X[test_index, :])

    X_test_all = X_test
    X_test_all_standardized = X_test_standardized

    # warm up the surrogate
    X_test_s = X_test_all[curr_models,]
    X_test_s_standardized = X_test_all_standardized[curr_models,]

    # load the performance evaluator f(.)
    clf = lgb.Booster(model_file=os.path.join(f'results/meta/{model_name}/PPE_{model_name}_full.txt'))

    f_pred = clf.predict(X_test_s)

    ######################
    # # train s on it -> selected train and the corresponding f results
    start = time.time()
    rfu = GaussianProcessRegressor()

    # only use HP configs.
    if isinstance(feature_index,int):
        rfu.fit(X_test_s_standardized[:, feature_index].reshape(-1,1), f_pred)
    else:
        rfu.fit(X_test_s_standardized[:, feature_index], f_pred)

    for epoch in range(n_epochs):
        # this should be changed to f's prediction
        curr_model_best_idx = np.argmax(f_pred)
        curr_model_best = np.max(f_pred)

        # update neighbor and weights for surrogate transfer
        neighbors, weights = get_sim_kendall(ind, f_pred, curr_models, n_neighbors, n_datasets, ap_all)

        # print(f"left_models: {left_models}")

        if isinstance(feature_index,int):
            mu_list, sigma_list = rfu.predict(X_test_all_standardized[left_models, :][:, feature_index].reshape(-1, 1), return_std=True)
        else:
            mu_list, sigma_list = rfu.predict(X_test_all_standardized[left_models, :][:, feature_index], return_std=True)

        neighbor_mu = []
        neighbor_sigma = []
        for n in neighbors:
            if isinstance(feature_index,int):
                mu_temp, sigma_temp = pre_clfs[n].predict(X_test_all_standardized[left_models, :][:, feature_index].reshape(-1, 1), return_std=True)
            else:
                mu_temp, sigma_temp = pre_clfs[n].predict(X_test_all_standardized[left_models, :][:, feature_index],
                                                        return_std=True)
            neighbor_mu.append(mu_temp)
            neighbor_sigma.append(sigma_temp)

        neighbor_mu_avg = average(np.asarray(neighbor_mu).T, weights.reshape(1, -1))

        mu_list = mu_list + neighbor_mu_avg

        z_list = (mu_list - curr_model_best) / sigma_list
        ei = (mu_list - curr_model_best) * norm.cdf(z_list) + sigma_list * norm.pdf(z_list)
        ei[np.where(sigma_list == 0)] = 0

        assert (len(ei) == len(left_models))

        # next best model
        ei_max = np.argmax(ei)
        next_model = left_models[ei_max]

        # refit the surrogate
        curr_models.append(next_model)
        left_models.remove(next_model)
        assert (len(curr_models) + len(left_models) == len(all_models))

        X_test_s = X_test_all[curr_models, :]
        X_test_s_standardized = X_test_all_standardized[curr_models, :]
        
        y_test_s = clf.predict(X_test_s)

        f_pred = y_test_s

        # retrain s on it -> selected train and the corresponding f results
        if isinstance(feature_index,int):
            rfu.fit(X_test_s_standardized[:, feature_index].reshape(-1,1), f_pred)
        else:
            rfu.fit(X_test_s_standardized[:, feature_index], f_pred)

        best_model_history[ind, epoch] = ap_all[ind, curr_models[curr_model_best_idx]]

        best_model_index[ind, epoch] = curr_models[curr_model_best_idx]

        ei_tracker[ind, epoch] = ei.max()

    # ours.append(ap_all[ind, curr_models[curr_model_best_idx]])
    # print('HPOD trail', ind, 'identified HP AP:', ours[-1])

    # XXX: I think the best hyperparameters can be found at 
    best_params = X_test_all[best_model_index[ind, epoch], feature_index]
    predicted_ap = f_pred[curr_model_best_idx]
    print(f'best hyperparams: {X_test_all[best_model_index[ind, epoch], feature_index]}')
    print(f'Predicted AP: {f_pred[curr_model_best_idx]}')
    
    if isinstance(feature_index,int):
        hyperparameters = pd.DataFrame.from_dict({"Orchard": f"Orchard {70+ind+1}", col_names[feature_index]: best_params, 'Predicted AP': predicted_ap}, orient='index').T
    elif model_name == 'IF':
        hyperparameters = pd.DataFrame.from_dict({"Orchard": f"Orchard {70+ind+1}", col_names[feature_index[0]]: best_params[0], col_names[feature_index[1]]: best_params[1], 'Predicted AP': predicted_ap}, orient='index').T    
    else:
        hyperparameters = pd.DataFrame.from_dict({"Orchard": f"Orchard {70+ind+1}", 
                                                  col_names[feature_index[0]]: best_params[0], 
                                                  col_names[feature_index[1]]: best_params[1], 
                                                  col_names[feature_index[2]]: best_params[2], 
                                                  col_names[feature_index[3]]: best_params[3],
                                                  col_names[feature_index[4]]: best_params[4],
                                                  'Predicted AP': predicted_ap}, orient='index').T
    OOS_params = pd.concat([OOS_params, hyperparameters], ignore_index=True)

OOS_params.to_csv(f"results/meta/Test/{model_name}_OOS_params.csv", index=False)


# ours_all = np.asarray(ours).reshape(1, n_datasets)
# ours = np.mean(ours_all, axis=0).tolist()

# ours_qth, ours_avg = process_batch(n_hp_configs, sorted_ap_val, ours_all, n_trials)

# print('*****************************************************************')
# print('Performance summary:')
# print('HPOD top-qth across 39 dataset', ours_qth)
# print('HPOD avg. normalized rank across 39 dataset', ours_avg)

# %%