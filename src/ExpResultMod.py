# %%
from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
import pdb
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

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


from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import utils.plotAnomaly as plotA
from sklearn.metrics import RocCurveDisplay

from Model import Geary

sampleSize = 4
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0


myData = utils.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
							  False)
data = myData.data.copy(deep=True)
delineations = myData.delineations.copy(deep=True)
mask = myData.mask.copy(deep=True)
spectralData = myData.spectralData
erf_num = myData.erf
refData = myData.ref_data.copy(deep=True)
# For plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255

# define the number of iterations
n_ite = 10
n_classifiers = 12

df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc',
              'ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD',
              'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD', 'Geary']


# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
ap_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(sampleSize):
    num = j

    myData = utils.engineer(num, 
                                data_paths_tif, 
                                data_paths_geojson, 
                                data_paths_geojson_zipped,
                                False)
    data = myData.data.copy(deep=True)
    delineations = myData.delineations.copy(deep=True)
    mask = myData.mask.copy(deep=True)
    spectralData = myData.spectralData
    erf_num = myData.erf
    refData = myData.ref_data.copy(deep=True)
    # For plotting
    tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    tryout = tryout/255

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
    roc_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]
    time_list = [erf_num, X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
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
        # TODO: use robust scaler in engineer class
        # X_train_norm, X_test_norm = standardizer(X_train, X_test)
        X_train_norm = utils.engineer._scaleData(X_train)
        X_test_norm = utils.engineer._scaleData(X_test)



# ECOD
#  INNE
        classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
            contamination=outliers_fraction),
            'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction),
            'Isolation Forest (IF)': IForest(contamination=outliers_fraction,
                                        random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
            'Minimum Covariance Determinant (MCD)': MCD(contamination=outliers_fraction),
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction),
            'Empirical Cumulative Distribution Functions (ECOD)': ECOD(contamination=outliers_fraction),
            'Kernal Principal Component Analysis (KPCA)': KPCA(
                contamination=outliers_fraction, random_state=random_state),
            'Isolation-based anomaly detection using nearest-neighbor ensembles (iNNE)': INNE(contamination=outliers_fraction),
            'Copula-Based Outlier Detection (COPOD)': COPOD(contamination=outliers_fraction),
            'Geary Multivariate Spatial Autocorrelation (Geary)': Geary(contamination=0.5, 
                                                                        geometry=data[["geometry"]], 
                                                                        centroid=data[["centroid"]])
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
            'Copula-Based Outlier Detection (COPOD)': 10,
            'Geary Multivariate Spatial Autocorrelation (Geary)': 11
        }

        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        count = 0
        for clf_name, clf in classifiers.items():
            t0 = time()
            if clf_name == 'Geary Multivariate Spatial Autocorrelation (Geary)':
                X1 = utils.engineer._scaleData(X)
                clf.fit(X1)
                test_scores = clf.decision_scores_
                y_test = y
            else:
                clf.fit(X_train_norm)
                test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            test_scores = np.nan_to_num(test_scores)

            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
            ap = round(average_precision_score(y_test, test_scores), ndigits=4)

            print('{clf_name} AUCROC:{roc}, precision @ rank n:{prn}, AP:{ap}, \
              execution time: {duration}s'.format(
                clf_name=clf_name, roc=roc, prn=prn, ap=ap, duration=duration))
            
            if j == sampleSize - 1:
                count += 1
                y_true = y_test
                y_pred = test_scores

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

                fig.savefig("test.png", dpi=fig.dpi)

            time_mat[i, classifiers_indices[clf_name]] = duration
            roc_mat[i, classifiers_indices[clf_name]] = roc
            prn_mat[i, classifiers_indices[clf_name]] = prn
            ap_mat[i, classifiers_indices[clf_name]] = ap

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    ap_list = ap_list + np.mean(ap_mat, axis=0).tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)

    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prc.csv', index=False, float_format='%.3f')
    ap_df.to_csv('ap.csv', index=False, float_format='%.3f')

# %%



t1 = time_df.copy(deep=True)
r1 = roc_df.copy(deep=True)
p1 = prn_df.copy(deep=True)
a1 = ap_df.copy(deep=True)


random_state = np.random.RandomState(i)

#         # 60% data for training and 40% for testing
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                             y,
#                                                             test_size=0.4,
#                                                             stratify=y,
#                                                             random_state=random_state)

# X_train_norm, X_test_norm = standardizer(X_train, X_test)


X = utils.engineer._scaleData(X)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
count = 0
plt.rcParams.update({'font.size': 15})
for clf_name, clf in classifiers.items():
    count += 1
    t0 = time()
    if clf_name == 'Geary Multivariate Spatial Autocorrelation (Geary)':
        clf.fit(X)
        test_scores = clf.decision_scores_
        y_test = y
    else:
        clf.fit(X)
        test_scores = clf.decision_function(X)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    test_scores = np.nan_to_num(test_scores)

    y_true = y_test
    y_pred = test_scores

    display = RocCurveDisplay.from_predictions(
            y_true,
            y_pred,
            pos_label=1,
            name=clf_name,
            ax=ax,
            plot_chance_level=(11 == count - 1),
            chance_level_kw={"linestyle": ":"},
            linewidth=5
        )

# fig.savefig("test.png", dpi=fig.dpi)

# %%
roc_df.T

# %%