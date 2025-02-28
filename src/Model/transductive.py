# %%
from __future__ import division
from __future__ import print_function

from time import time
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import utils
import numpy as np
import pandas as pd

from Model import Geary

import pickle

def transformData(data, PR=False):
    # y is either precision or TPR
    y = pd.DataFrame(data[0])
    # x is either recall or FPR
    x = pd.DataFrame(data[1])
    # result is either AUC or AP
    result = pd.DataFrame.from_dict({ keys: str(i) for keys, i in data[2].items() }, orient='index')

    if PR == False:
        y = y.melt(var_name='Estimator', value_name='TPR')
        x = x.melt(var_name='Estimator', value_name='FPR')
    else:
        y = y.melt(var_name='Estimator', value_name='Precision')
        x = x.melt(var_name='Estimator', value_name='Recall')

    df = pd.concat([y, x], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    df['Type'] = None

    for i in df["Estimator"].unique():
        if i == 'ECOD':
            df.loc[df['Estimator'] == i, 'Type'] = 'Probabilistic'
        elif i == 'IF':
            df.loc[df['Estimator'] == i, 'Type'] = 'Learning'
        elif i == 'PCA':
            df.loc[df['Estimator'] == i, 'Type'] = 'Reconstruction'
        elif i == 'LOF':
            df.loc[df['Estimator'] == i, 'Type'] = 'Density'
        elif i == 'ABOD':
            df.loc[df['Estimator'] == i, 'Type'] = 'Distance'
        elif i == 'Geary':
            df.loc[df['Estimator'] == i, 'Type'] = 'Spatial'

    result.reset_index(inplace=True)
    if PR == False:
        result.columns = ['Estimator','AUC']
    else:
        result.columns = ['Estimator','AP']

    return (df, result)

def getROC(y_true, y_pred, mean_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    tpr = np.interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0

    auc = roc_auc_score(y_true, y_pred)

    return tpr, mean_fpr, round(auc, ndigits=3)

def getPR(y_true, y_pred, mean_recall):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = auc(recall, precision)
    
    # precision = np.interp(mean_recall, recall, precision)
    # precision[0] = 1.0

    return precision, recall, round(ap, ndigits=3) #mean_recall, round(ap, ndigits=3)

    # TODO: Pick outliers factor that is common over many orchards as a default!
def estimators(outliers_fraction, params, geometry=None, centroid=None):
    return {
        'ABOD': 
            ABOD( contamination=outliers_fraction, 
                 n_neighbors=params['ABOD']['n_neighbors']
                 ),
                
        'IF': 
            IForest(contamination=outliers_fraction,
                    n_estimators=params['IF']['n_estimators'],
                    max_features=params['IF']['max_features'],
                    behaviour='new',
                    n_jobs=5,
                    random_state=42),
                        
        'LOF': 
            LOF(contamination=outliers_fraction,
                n_neighbors=params['LOF']['n_neighbors'],
                metric=params['LOF']['metric']
                ),

        'ECOD': 
            ECOD(contamination=outliers_fraction),

        'PCA':
            PCA(contamination=outliers_fraction,
                n_selected_components=params['PCA']['n_selected_components']
                ),

        'Geary':
            Geary(contamination=outliers_fraction, 
                  geometry=geometry, 
                  centroid=centroid)
        }

def indices():
    return {
        'ABOD': 0,
        'IF': 1,
        'LOF': 2,
        'ECOD': 3,
        'PCA':4,
        'Geary': 5
    }

def init_results(keys, pop_size):
    # true positive rate, false positive rate, thresholds
    return {key: np.zeros([pop_size]) for key in keys}, {key: np.zeros([pop_size]) for key in keys}, {key: 0 for key in keys}


def transductionResults(data, erf_num, localOF, angleBOD, principleCA, isolationF):
    n_classifiers = 6

    df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc %',
                'ABOD', 'IForest', 'LOF', 'ECOD', 'PCA', 'Geary']

    # initialize the container for saving the results
    aucroc_df = pd.DataFrame(columns=df_columns)
    ap_df = pd.DataFrame(columns=df_columns)
    time_df = pd.DataFrame(columns=df_columns)

    X = np.array(data.loc[:, "confidence":])
    y = np.array(data.loc[:, "Y"]).T 
    # Change outlier to 1 and inlier to 0 in data
    y = np.where(y == 'Outlier', 1, 0)

    outliers_fraction = np.count_nonzero(y) / len(y)
    if outliers_fraction > 0:
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)
    else:
        outliers_percentage = 0.1
    
    localOF["contamination"] = outliers_fraction
    angleBOD["contamination"] = outliers_fraction
    principleCA["contamindation"] = outliers_fraction
    isolationF["contamination"] = outliers_fraction


    params = {'LOF': localOF,
              'ABOD': angleBOD, 
              'PCA': principleCA, 
              'IF': isolationF}

    # construct containers for saving results
    aucroc_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]
    time_list = ["Orchard_"+erf_num, X.shape[0], X.shape[1], outliers_percentage]

    # initialize the container for saving the results
    aucroc_mat = np.zeros([n_classifiers])
    ap_mat = np.zeros([n_classifiers])
    time_mat = np.zeros([n_classifiers])

    fpr_pop = 100
    mean_fpr = np.linspace(0, 1, fpr_pop)

    # just to give the indices a label.
    classifiers_indices = indices()

    # TODO: create a dictionary to hold matrix of tpr, fpr, thresholds results for 
    # each classifier and each iteration
    tpr_results, fpr_results, auc = init_results(keys=classifiers_indices.keys(),
                                                 pop_size=fpr_pop)#y.shape[0])
    precision, recall, aucpr = init_results(keys=classifiers_indices.keys(),
                                            pop_size=fpr_pop)

    labels = tpr_results.copy()

    random_state = np.random.RandomState(42)

    # classifiers must be reinitialized for each iteration
    classifiers = estimators(outliers_fraction, 
                             params,
                             data["geometry"], 
                             data["centroid"])

    # standardizing data for processing
    X = utils.engineer._scaleData(X)

    for clf_name, clf in classifiers.items():
        t0 = time()
        clf.fit(X)
        test_scores = clf.decision_scores_
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        test_scores = np.nan_to_num(test_scores)
        
        # for now we assume that average precision is the area under the precision recall curve
        # https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b

        aucroc = round(roc_auc_score(y, test_scores), ndigits=3)
        ap = round(average_precision_score(y, test_scores), ndigits=3)
        
        print('{clf_name} AUCROC:{aucroc}, AP:{ap}, \
        execution time: {duration}s'.format(
            clf_name=clf_name, aucroc=aucroc, ap=ap, duration=duration))
        
        # tpr_results[clf_name], fpr_results[clf_name], _ = roc_curve(y, test_scores)
        tpr_results[clf_name], fpr_results[clf_name], auc[clf_name] = getROC(y, test_scores, mean_fpr)
        
        precision[clf_name], recall[clf_name], aucpr[clf_name] = getPR(y, test_scores, mean_fpr[::-1])
        # auc[clf_name] = aucroc

        labels[clf_name] = clf.labels_

        time_mat[classifiers_indices[clf_name]] = duration
        aucroc_mat[classifiers_indices[clf_name]] = aucroc
        ap_mat[classifiers_indices[clf_name]] = ap

    time_list = time_list + time_mat.tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    aucroc_list = aucroc_list + aucroc_mat.tolist()
    temp_df = pd.DataFrame(aucroc_list).transpose()
    temp_df.columns = df_columns
    aucroc_df = pd.concat([aucroc_df, temp_df], axis=0)

    ap_list = ap_list + ap_mat.tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)
    
    output = transformData((tpr_results, fpr_results, auc))
    output = output + (labels,)
    output = output + (precision, recall, aucpr)

    
    pickle.dump(output,
                open("results/transductive/" + erf_num + ".pkl", "wb"))

    return aucroc_df, ap_df, time_df
