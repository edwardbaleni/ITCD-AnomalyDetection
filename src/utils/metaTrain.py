import numpy as np
import pandas as pd
import joblib
from os import listdir
from sklearn.preprocessing import RobustScaler
from utils.IPM import IPM
from utils.gen_meta_features import generate_meta_features
from sklearn.preprocessing import RobustScaler 
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import joblib

import lightgbm as lgb
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelBinarizer 

# TODO: put all the parts in functions and hide away!

# TODO: Load in Performance matrix, P
    # This is obtained from hyper
    # this only happens on training data
def loadPerformance(model_name):

    data = joblib.load("results/training/data0_70.pkl")

    # we are not looking at datasets 21, 49, 70
    indices_to_delete = [20, 48, 69]
    data = data[0:20] + data[21:48] + data[49:69]

    
    path = "results/hyperparameter/"
    home = listdir(path)

    # either ABOD, LOF, PCA, or IF
    files = [f'{path}{file}' for file in home if model_name in file]

    study = [joblib.load(file) for file in files]

    perf = pd.DataFrame()
    HPs = []
    for i in range(len(study)):
        trials = study[i].trials_dataframe()
        trials['Dataset'] = i
        if model_name == 'LOF':    
            perf = pd.concat([perf, trials.loc[:, ['Dataset', 'value', 'params_n_neighbors', 'params_metric']]])
            HPs = ['params_n_neighbors', 'params_metric']
        elif model_name == 'ABOD':
            perf = pd.concat([perf, trials.loc[:, ['Dataset', 'value', 'params_n_neighbors']]])
            HPs = ['params_n_neighbors']
        elif model_name == 'PCA':
            perf = pd.concat([perf, trials.loc[:, ['Dataset', 'value', 'params_n_selected_components']]])
            HPs = ['params_n_selected_components']
        elif model_name == 'IF':
            perf = pd.concat([perf, trials.loc[:, ['Dataset', 'value', 'params_max_features', 'params_n_estimators']]])
            HPs = ['params_max_features', 'params_n_estimators']


    for i in range(len(data)):
        data[i].loc[:,"confidence":] = RobustScaler().fit_transform(data[i].loc[:, "confidence":])

    return perf, data, HPs, study

# Load in internal Performance, IPM measures.
    # Use the results from tuning to get O 
    # Use O to get the IPM measures
    # Save the IPM measures
def getIPM(perf, data, model_name):

    outlier_score = []

    from pyod.models.lof import LOF
    from pyod.models.abod import ABOD
    from pyod.models.pca import PCA
    from pyod.models.iforest import IForest

    for i in set(perf['Dataset']):
        output = perf[perf['Dataset'] == i]
        scores = pd.DataFrame()
        for j in range(output.shape[0]):
            # get the hyperparameters at the jth index
            if model_name == 'LOF':
                neighbour = (output['params_n_neighbors'].iloc[j])
                metric = (output['params_metric'].iloc[j])
                clf = LOF(n_neighbors=neighbour, metric=metric)
            elif model_name == 'ABOD':
                neighbour = (output['params_n_neighbors'].iloc[j])
                clf = ABOD(n_neighbors=neighbour)
            elif model_name == 'PCA':
                n_components = (output['params_n_selected_components'].iloc[j])
                clf = PCA(n_components=n_components)
            elif model_name == 'IF':
                max_features = (output['params_max_features'].iloc[j])
                n_estimators = (output['params_n_estimators'].iloc[j])
                clf = IForest(max_features=max_features, n_estimators=n_estimators)
            
            clf.fit(data[i].loc[:, "confidence":])
            y_test_scores = clf.decision_scores_
            
            if model_name == 'LOF':
                scores[f"(LOF, {(neighbour, metric)})"] = y_test_scores
            elif model_name == 'ABOD':
                scores[f"(ABOD, {(neighbour)})"] = y_test_scores
            elif model_name == 'PCA':
                scores[f"(PCA, {(n_components)})"] = y_test_scores
            elif model_name == 'IF':
                scores[f"(IF, {(max_features, n_estimators)})"] = y_test_scores

        
        print(f'Orchard {i+1} done')
        outlier_score.append(scores)

    MC, SELECT_s, SELECT_t, HITS_s, HITS_t = IPM(outlier_score, data, train=True)
    return MC, SELECT_s, SELECT_t, HITS_s, HITS_t, output

# extract meta-features per task g(X)
def getMetaFeatures(data, output, perf, HPs, study, model_name, MC, SELECT_s, HITS_s):
    _, feats = generate_meta_features(data[0].loc[:, "confidence":])

    feature_list = feats+HPs+['MC', 'SELECT', 'HITS']
    meta_dataframe = pd.DataFrame(columns=feature_list)

    # Binarize the 'params_metric' column
    if model_name == 'LOF':
        lb = LabelBinarizer()
        binarize = lb.fit_transform(output['params_metric'])

        metric_df = pd.DataFrame(binarize, 
                                columns = lb.classes_) 
        
    # 0 isnt working very well right now
    # TODO: Change the 38 to len(data)
    for i in range(len(study)):
        meta_vals, meta_feats = generate_meta_features(data[i].loc[:, "confidence":])
        meta_vals = np.nan_to_num(meta_vals, copy=False)

        meta_vals = np.tile(meta_vals.T, MC[i].shape[0])
        meta_vals = meta_vals.reshape(MC[i].shape[0], len(meta_feats))

        # HP = [int(re.findall(r'\d+', j)[0]) for j in outlier_score[i].columns]
        if model_name == 'LOF':
            meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                                # pd.DataFrame({'params_n_neighbors': HP}),
                                            output.loc[:, ['params_n_neighbors']],
                                            metric_df,
                                            pd.DataFrame({'MC'    : MC[i]}), 
                                            pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                            pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
        elif model_name == 'ABOD':
            meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                            output.loc[:, ['params_n_neighbors']],
                                            pd.DataFrame({'MC'    : MC[i]}), 
                                            pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                            pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
        elif model_name == 'PCA':
            meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                            output.loc[:, ['params_n_selected_components']],
                                            pd.DataFrame({'MC'    : MC[i]}), 
                                            pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                            pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
        elif model_name == 'IF':
            meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                            output.loc[:, ['params_max_features', 'params_n_estimators']],
                                            pd.DataFrame({'MC'    : MC[i]}), 
                                            pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                            pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
        meta_['Dataset'] = i
        
        meta_dataframe = pd.concat([meta_dataframe, meta_])

    y_meta = pd.DataFrame()
    cols = ['Dataset'] + [col for col in meta_.columns if col != 'Dataset']
    meta_dataframe = meta_dataframe[cols]

    for i in set(perf['Dataset']):
        output = perf[perf['Dataset'] == i]

        # get per row of the dataset
        for _, row in meta_dataframe[meta_dataframe["Dataset"] == i].iterrows():
            if model_name == 'LOF' or model_name == 'ABOD':
                ind = np.where(output['params_n_neighbors'] == row['params_n_neighbors'])
            elif model_name == 'PCA':
                ind = np.where(output['params_n_selected_components'] == row['params_n_selected_components'])
            elif model_name == 'IF':
                ind = np.where((output['params_max_features'] == row['params_max_features']) & 
                                (output['params_n_estimators'] == row['params_n_estimators']))
            ind = ind[0][0]
            val = output['value'].iloc[[ind]]
            y_meta = pd.concat([y_meta, val])
        

    meta_dataframe.reset_index(drop=True, inplace=True)
    if model_name == 'LOF' or model_name == 'ABOD':
        meta_dataframe['params_n_neighbors'] = meta_dataframe['params_n_neighbors'].astype(int)
    elif model_name == 'PCA':
        meta_dataframe['params_n_selected_components'] = meta_dataframe['params_n_selected_components'].astype(int)
    elif model_name == 'IF':
        meta_dataframe['params_max_features'] = meta_dataframe['params_max_features'].astype(int)
        meta_dataframe['params_n_estimators'] = meta_dataframe['params_n_estimators'].astype(int)

    meta_dataframe.drop(columns=['Dataset'], inplace=True)
    y_meta.reset_index(drop=True, inplace=True)

    return meta_dataframe, y_meta

# Train the proxy performance evaluator f(.)
def objective(trial, X, Y):
    params = {'metric': 'rmse',
                'objective': 'regression',
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
                # 'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
                # 'max_depth': trial.suggest_int('max_depth', 1, 32),
                # 'num_iterations': trial.suggest_int('num_iterations', 1, 1000),
                'num_threads': 5,
                'num_leaves': trial.suggest_int('num_leaves', 2, 64),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 15, 30),

                'boosting_type': 'gbdt',
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'verbose': -1
                }
    
    return train(params, X, Y)

def train(params, X, Y):
    """
    Trains the given model using a transductive approach and calculates the average precision score.
    Args:
        model: The anomaly detection model to be trained. It should have a `fit` method and a `decision_scores_` attribute.
    Returns:
        float: The average precision score of the model on the training data.
    """
    cv_results = lgb.cv(params, 
                        lgb.Dataset(X, Y), 
                        nfold=5, 
                        # num_boost_round=5000, 
                        stratified=False)
    
    return np.mean(cv_results['valid rmse-mean'])

def trainPPE(data, meta_dataframe, y_meta, model_name):
    kf = KFold(n_splits=len(data))
    ind = -1

    # We train the entire PPE on all the data available
    # so it is like LOO-CV where we train on all but one
    for train_index, test_index in kf.split(meta_dataframe):
        # print(test_index)
        ind += 1
        # XXX: to not repeat trials
        if (ind < 45 and model_name == 'LOF') or (ind < 23 and model_name == 'ABOD') or (ind < 17 and model_name == 'IF'): 
            continue
        
        X = meta_dataframe.iloc[train_index]
        y = y_meta.iloc[train_index]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        func = lambda trial: objective(trial, X_train, y_train)
        study = optuna.create_study()

        study.optimize(func, n_trials=500)

        # Test model
        params = study.best_params
        params['num_threads'] = 5
        params['metric'] = 'rmse'
        params['objective'] = 'regression'
        params['boosting_type'] = 'gbdt'
        params['verbose'] = -1
        num_round = 10

        bst = lgb.train(params, lgb.Dataset(X, y))
        bst.save_model(f'results/meta/{model_name}/PPE_{model_name}_{ind}.txt')

        

# full dataset
def train_full(meta_dataframe, y_meta, model_name):
    X = meta_dataframe
    y = y_meta
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    func = lambda trial: objective(trial, X_train, y_train)
    study = optuna.create_study()

    study.optimize(func, n_trials=500)

    # Test model
    params = study.best_params
    params['num_threads'] = 5
    params['metric'] = 'rmse'
    params['objective'] = 'regression'
    params['boosting_type'] = 'gbdt'
    params['verbose'] = -1
    num_round = 10

    bst = lgb.train(params, lgb.Dataset(X, y))
    bst.save_model(f'results/meta/{model_name}/PPE_{model_name}_full.txt')