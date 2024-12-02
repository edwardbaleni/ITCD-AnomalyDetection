# %%
from __future__ import division
from __future__ import print_function

from time import time
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.kpca import KPCA
from pyod.models.ecod import ECOD
from pyod.models.inne import INNE

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay

import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA

from Model import Geary

import pickle

# def plotROC(y_true, y_pred, clf_name, fig, ax, count):
#     display = RocCurveDisplay.from_predictions(
#         y_true,
#         y_pred,
#         pos_label=1,
#         name=clf_name,
#         ax=ax,
#         plot_chance_level=( count - 1 == 11),
#         chance_level_kw={"linestyle": ":"},
#         linewidth=5
#         )

def getAverageROC(y_true, y_pred, mean_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    roc_auc = roc_auc_score(y_true, y_pred)

    return mean_tpr, mean_fpr, roc_auc
    
def estimators(outliers_fraction, random_state):
    return {
        'Angle-based Outlier Detector (ABOD)': 
            ABOD(contamination=outliers_fraction),
            
        'Cluster-based Local Outlier Factor (CBLOF)': 
            CBLOF(n_clusters=10,
                    contamination=outliers_fraction,
                    check_estimator=False,
                    random_state=random_state),
    
        'Histogram-base Outlier Detection (HBOS)': 
            HBOS(contamination=outliers_fraction),
                
        'Isolation Forest (IF)': 
            IForest(contamination=outliers_fraction,
                    random_state=random_state),
                        
        'K Nearest Neighbors (KNN)': 
            KNN(contamination=outliers_fraction),
                
        'Minimum Covariance Determinant (MCD)': 
            MCD(contamination=outliers_fraction),
    
        'Local Outlier Factor (LOF)': 
            LOF(contamination=outliers_fraction),

        'Empirical Cumulative Distribution Functions (ECOD)': 
            ECOD(contamination=outliers_fraction),
    
        'Kernal Principal Component Analysis (KPCA)': 
            KPCA(contamination=outliers_fraction, random_state=random_state),

        'Isolation-based anomaly detection using nearest-neighbor ensembles (iNNE)': 
            INNE(contamination=outliers_fraction),
                
        'Copula-Based Outlier Detection (COPOD)': 
            COPOD(contamination=outliers_fraction)
        }

def indices():
    return {
        'Angle-based Outlier Detector (ABOD)': 0,
        'Cluster-based Local Outlier Factor (CBLOF)': 1,
        'Histogram-base Outlier Detection (HBOS)': 2,
        'Isolation Forest (IF)': 3,
        'K Nearest Neighbors (KNN)': 4,
        'Minimum Covariance Determinant (MCD)': 5,
        'Local Outlier Factor (LOF)': 6,
        'Empirical Cumulative Distribution Functions (ECOD)': 7,
        'Kernal Principal Component Analysis (KPCA)': 8,
        'Isolation-based anomaly detection using nearest-neighbor ensembles (iNNE)': 9,
        'Copula-Based Outlier Detection (COPOD)': 10
    }

def init_results(keys, experiments, pop_size):
    # true positive rate, false positive rate, thresholds
    return {key: np.zeros([experiments, pop_size]) for key in keys}, {key: np.zeros([experiments, pop_size]) for key in keys}, {key: np.zeros([experiments]) for key in keys}

def init_mean_results(keys, pop_size):
    # true positive rate, false positive rate, thresholds
    return {key: np.zeros([pop_size]) for key in keys}, {key: np.zeros([pop_size]) for key in keys}, {key: 0 for key in keys}, {key: 0 for key in keys}


def inductionResults(data, erf_num):
    # define the number of iterations
    n_ite = 5
    n_classifiers = 11

    df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc %',
                'ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD',
                'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD']


    # initialize the container for saving the results
    aucroc_df = pd.DataFrame(columns=df_columns)
    ap_df = pd.DataFrame(columns=df_columns)
    time_df = pd.DataFrame(columns=df_columns)

    # placeholder to hold the results
    fpr_pop = 200
    mean_fpr = np.linspace(0, 1, fpr_pop)

    X = np.array(data.loc[:, "confidence":])
    y = np.array(data.loc[:, "Y"]).T 
    # Change outlier to 1 and inlier to 0 in data
    y = np.where(y == 'Outlier', 1, 0)

    outliers_fraction = np.count_nonzero(y) / len(y)
    if outliers_fraction > 0:
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)
    else:
        outliers_percentage = 0.1

    # construct containers for saving results
    aucroc_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]
    time_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]

    aucroc_mat = np.zeros([n_ite, n_classifiers])
    ap_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    # just to give the indices a label.
    classifiers_indices = indices()

    # TODO: create a dictionary to hold matrix of tpr, fpr, thresholds results for 
    # each classifier and each iteration
    tpr_results, fpr_results, auc  = init_results(keys=classifiers_indices.keys(), 
                                         experiments= n_ite, 
                                         pop_size= fpr_pop)#len(y) )
    
    mean_tpr_out, mean_fpr_out, mean_auc_out, std = init_mean_results(keys=classifiers_indices.keys(), 
                                                                 pop_size= fpr_pop)
    
    std_auc = mean_auc_out.copy()
    tpr_std = mean_auc_out.copy()
    tpr_upper = mean_tpr_out.copy()
    tpr_lower = mean_tpr_out.copy()
    

    for i in range(n_ite):
        print("\n... Processing", erf_num, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # classifiers must be reinitialized for each iteration
        classifiers = estimators(outliers_fraction, random_state)

        # 60% data for training and 40% for testing
        X_train, X_test, _, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.4,
                                                      stratify=y,
                                                      random_state=random_state)

        # standardizing data for processing
        X_train_norm = utils.engineer._scaleData(X_train)
        X_test_norm = utils.engineer._scaleData(X_test)


        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            test_scores = np.nan_to_num(test_scores)
            
            # for now we assume that average precision is the area under the precision recall curve
            # https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b

            aucroc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            ap = round(average_precision_score(y_test, test_scores), ndigits=4)
            
            print('{clf_name} AUCROC:{aucroc}, AP:{ap}, \
            execution time: {duration}s'.format(
                clf_name=clf_name, aucroc=aucroc, ap=ap, duration=duration))
            
            tpr_results[clf_name][i], fpr_results[clf_name][i], auc[clf_name][i] = getAverageROC(y_test, test_scores, mean_fpr)

            

            time_mat[i, classifiers_indices[clf_name]] = duration
            aucroc_mat[i, classifiers_indices[clf_name]] = aucroc
            ap_mat[i, classifiers_indices[clf_name]] = ap

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    aucroc_list = aucroc_list + np.mean(aucroc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(aucroc_list).transpose()
    temp_df.columns = df_columns
    aucroc_df = pd.concat([aucroc_df, temp_df], axis=0)

    ap_list = ap_list + np.mean(ap_mat, axis=0).tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)

    # Save mean tpr, fpr, auc, std for each classifier
    for key in tpr_results.keys():
        mean_tpr_out[key] = np.mean(tpr_results[key], axis=0)
        mean_fpr_out[key] = mean_fpr
        mean_auc_out[key] = np.mean(auc[key])
        std_auc[key] = np.std(auc[key])
        tpr_std[key] = np.std(tpr_results[key], axis=0)
        tpr_upper[key] = np.minimum(mean_tpr_out[key] + std[key], 1)
        tpr_lower[key] = mean_tpr_out[key] - std[key]
    
    pickle.dump((mean_tpr_out, mean_fpr_out, mean_auc_out, tpr_std, tpr_upper, tpr_lower, std_auc),
                open("results/inductive/" + erf_num + ".pkl", "wb"))

    return (aucroc_df.T, ap_df.T, time_df.T)