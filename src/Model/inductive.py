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
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay

import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA

from Model import Geary

def plotROC(y_true, y_pred, clf_name, fig, ax, count):
    display = RocCurveDisplay.from_predictions(
        y_true,
        y_pred,
        pos_label=1,
        name=clf_name,
        ax=ax,
        plot_chance_level=( count - 1 == 11),
        chance_level_kw={"linestyle": ":"},
        linewidth=5
        )

def inductionResults(data, erf_num):
    # define the number of iterations
    n_ite = 10
    n_classifiers = 11

    df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc %',
                'ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD',
                'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD']


    # initialize the container for saving the results
    aucroc_df = pd.DataFrame(columns=df_columns)
    ap_df = pd.DataFrame(columns=df_columns)
    time_df = pd.DataFrame(columns=df_columns)

    mat_file_list = erf_num
    mat = data
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
    aucroc_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]
    time_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]

    aucroc_mat = np.zeros([n_ite, n_classifiers])
    ap_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", erf_num, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.4,
                                                            stratify=y,
                                                            random_state=random_state)

        # standardizing data for processing
        X_train_norm = utils.engineer._scaleData(X_train)
        X_test_norm = utils.engineer._scaleData(X_test)

        classifiers = {'Angle-based Outlier Detector (ABOD)': 
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

        classifiers_indices = {
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

        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        count = 0
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
            
            if i == (n_ite - 1):
                count += 1
                plotROC(y_test, test_scores, clf_name, fig, ax, count)
                fig.savefig("test.png", dpi=fig.dpi)

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



    return (aucroc_df.T, ap_df.T, time_df.T)