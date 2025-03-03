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


# perform outlier detection
for i in range(len(data)):
    scaler = RobustScaler()
    dataset = scaler.fit_transform(data[i].loc[:,'confidence':])
    # LOF
    clf = LOF(n_neighbors=int(lof['params_n_neighbors'][i]), metric=lof['metric'][i])
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    plotScores(images[i], data[i], y_test_scores, f"results/oos/outliers/lof_{i+71}.png")
    plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/out_scores/lof_{i+71}.png")


    # # ABOD
    clf = ABOD(n_neighbors=abod['params_n_neighbors'][i])
    # Isolation Forest
    clf = IForest(n_estimators=int(iforest['params_n_estimators'][i]), max_features=int(iforest['params_max_features'][i]))
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    plotScores(images[i], data[i], y_test_scores, f"results/oos/outliers/iforest_{i+71}.png")
    plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/out_scores/iforest_{i+71}.png")

    # # PCA
    clf = PCA(n_selected_components=int(pca['params_n_selected_components'][i]))
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    plotScores(images[i], data[i], y_test_scores, f"results/oos/outliers/pca_{i+71}.png")
    plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/out_scores/pca_{i+71}.png")

    # # ECOD
    clf = ECOD()
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    plotScores(images[i], data[i], y_test_scores, f"results/oos/outliers/ecod_{i+71}.png")
    plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/out_scores/ecod_{i+71}.png")

    # # Geary
    clf = Geary(contamination=0.1,
                geometry=data[i]["geometry"], 
                centroid=data[i]["centroid"])
    clf.fit(dataset)
    y_test_scores = clf.decision_scores_
    y_labels = clf.labels_
    plotScores(images[i], data[i], y_test_scores, f"results/oos/outliers/geary_{i+71}.png")
    plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/out_scores/geary_{i+71}.png")

    # # XXX: Maybe
    # # obtain ROC and PR curves
    # # obtain AUC and AP



# %%