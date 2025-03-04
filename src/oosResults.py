# %%
import joblib
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from Model import Geary

from utils.plotAnomaly import plotScores, plot


from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import roc_auc_score

# %%

data = joblib.load("results/testing/data70_101.pkl")
images = joblib.load("results/testing/images70_101.pkl")

# %%

#     img = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
#     img = img/255

# need to donwload masks and images

lof = pd.read_csv("results/meta/Test/LOF_OOS_params.csv")
abod = pd.read_csv("results/meta/Test/ABOD_OOS_params.csv")
iforest = pd.read_csv("results/meta/Test/IF_OOS_params.csv")
pca = pd.read_csv("results/meta/Test/PCA_OOS_params.csv")
# Geary
# ECOD

# Melt the dataframe and drop NA values
melted_lof = lof.melt(id_vars=['Orchard', 'Predicted AP', 'params_n_neighbors'],
                      value_vars=['chebyshev', 'euclidean', 'manhattan', 'minkowski']).dropna()
# Filter out rows where the value is 0 and reset the index
melted_lof.rename(columns={'variable': 'metric'}, inplace=True)
lof = melted_lof[melted_lof['value'] != 0].reset_index(drop=True)
del lof['value']

ap_scores = pd.DataFrame()
auc_scores = pd.DataFrame()


# perform outlier detection
for i in range(len(data)):
    scaler = RobustScaler()
    dataset = scaler.fit_transform(data[i].loc[:,'confidence':])
    y = np.array(data[i].loc[:, "Y"]).T 
    y = np.where(y == 'Outlier', 1, 0)
    # LOF
    clf = LOF(contamination=0.05, n_neighbors=int(lof['params_n_neighbors'][i]), metric=lof['metric'][i])
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_

    ap_lof = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        auc_lof = 0
    else:
        auc_lof = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/lof_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/lof_{i+71}.png")
    print(f'LOF - AP: {ap_lof}, Predicted AP: {lof["Predicted AP"][i]}, AUC: {auc_lof}')


    # # ABOD
    clf = ABOD(contamination=0.05, n_neighbors=int(abod['params_n_neighbors'][i]))
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    abod_ap = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        abod_auc = 0
    else:
        abod_auc = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/abod_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/abod_{i+71}.png")
    print(f'ABOD - AP: {abod_ap}, Predicted AP: {abod["Predicted AP"][i]}, AUC: {abod_auc}')


    # Isolation Forest
    clf = IForest(n_estimators=int(iforest['params_n_estimators'][i]), max_features=int(iforest['params_max_features'][i]))
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    iforest_ap = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        iforest_auc = 0
    else:
        iforest_auc = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/iforest_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/iforest_{i+71}.png")
    print(f'Iforest - AP: {iforest_ap}, Predicted AP: {abod["Predicted AP"][i]}, AUC: {abod_auc}')

    # # PCA
    clf = PCA(contamination=0.05, n_selected_components=int(pca['params_n_selected_components'][i]))
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    pca_ap = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        pca_auc = 0
    else:
        pca_auc = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/pca_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/pca_{i+71}.png")
    print(f'PCA - AP: {average_precision_score(y, y_test_scores)}, Predicted AP: {pca["Predicted AP"][i]}, AUC: {pca_auc}')


    # # ECOD
    clf = ECOD(contamination=0.05)
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    ecod_ap = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        ecod_auc = 0
    else:
        ecod_auc = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/ecod_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/ecod_{i+71}.png")
    print(f'ECOD - AP: {ecod_ap}', f'AUC: {ecod_auc}')


    # # Geary
    clf = Geary(contamination=0.05,
                geometry=data[i]["geometry"], 
                centroid=data[i]["centroid"])
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    geary_ap = average_precision_score(y, y_test_scores)
    if np.sum(y) == 0:
        geary_auc = 0
    else:
        geary_auc = roc_auc_score(y, y_test_scores)
    # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/geary_{i+71}.png")
    # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/geary_{i+71}.png")
    print(f'Geary - AP: {geary_ap}, AUC: {geary_auc}')    
    
    # # obtain AUC and AP
    auc_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": auc_lof, "ABOD": abod_auc, "IF": iforest_auc, "PCA": pca_auc, "ECOD": ecod_auc, "Geary": geary_auc}, orient='index').T
    auc_scores = pd.concat([auc_scores, auc_data])
    ap_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": ap_lof, "ABOD": abod_ap, "IF": iforest_ap, "PCA": pca_ap, "ECOD": ecod_ap, "Geary": geary_ap}, orient='index').T
    ap_scores = pd.concat([ap_scores, ap_data])

    # # XXX: Maybe
    # # obtain ROC and PR curves




# %%
ap_scores.reset_index(drop=True, inplace=True)
auc_scores.reset_index(drop=True, inplace=True)
ap_scores.to_csv("results/oos/ap_scores.csv")
auc_scores.to_csv("results/oos/auc_scores.csv")


# %%
# XXX: Interpretability!
from esda import Geary_Local
import utils.Triangulation as tri
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd


data2 = joblib.load("results/testing/data70_101.pkl")

data_local = data2[1]
geometry = data_local["geometry"]
centroid = data_local["centroid"]

scaler = RobustScaler()
data_local.loc[:,'confidence':] = scaler.fit_transform(data_local.loc[:, 'confidence':])

ww , _, _, _ = tri.delauneyTriangulation(pd.concat([geometry, centroid], axis=1))

cols = list(data_local.columns)[5:]
uni_Geary = pd.DataFrame()
for i in range(len(cols)):

    X = data_local.loc[:,cols[i]]
    lG_mv = Geary_Local(connectivity=ww).fit(X)
    
    # Transform scores
    centerScore = lG_mv.localG - np.mean(lG_mv.localG)
    probs = expit(centerScore)
    uni_Geary[cols[i]] = probs

clf = Geary(contamination=0.05,
                geometry=data_local["geometry"], 
                centroid=data_local["centroid"])
X = np.array(data_local.loc[:, 'confidence':])
clf.fit(X)
y_test_scores = clf.decision_scores_
y_labels = clf.labels_


outliers = uni_Geary[y_labels == 1]
# outliers.reset_index(drop=True, inplace=True)
import matplotlib.pyplot as plt


# Plot each observation (y-axis) versus the dimensions (x-axis)
# First calculate the 99th percentile for each dimension
percentile_99 = outliers.quantile(0.99, axis=0)

# Plot individual observations and the 99th percentile line
for idx in outliers.index:
    plt.figure(figsize=(12, 6))
    # Plot individual observation
    plt.plot(outliers.columns, outliers.loc[idx, :], marker='o', linestyle='-', alpha=0.7, label=f'Observation {idx}')
    
    # Plot 99th percentile line
    plt.plot(outliers.columns, percentile_99, marker='s', linestyle='--', color='red', linewidth=2, label='99th percentile')
    
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Observation {idx} vs 99th Percentile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/oos/interpretability/demo_obs_{idx}.png")
    plt.show()



    




# %%
