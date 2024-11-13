# %%
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import utils.plotAnomaly as plotA

sampleSize = 20
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
# For plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255

# %%

# Below methods are in line with: https://ieeexplore-ieee-org.ezproxy.uct.ac.za/document/9297055
# Paper says that isolation forests are the best option for AD

# Refernce:
    # https://arxiv.org/pdf/2206.09426
    # https://pyod.readthedocs.io/en/latest/index.html

# removing distances is helpful as trees on the edge that are correct
# are sometimes anomalies when they should not be
# data.drop(['dist1', 'dist2', 'dist3', 'dist4'], axis = 1, inplace=True)

# These features don't seem to improve AD at all
# data.drop(["crown_projection_area", "crown_perimeter", "radius_of_gyration", "minor_axis", "major_axis"], axis = 1, inplace=True)
# %%

	# when looking at colour specs only in AD, there is a problem. 
	# It is not simply just between soil and crown
	# we also start to pick up ill trees
	# trees that may be burnt by sun
	# and different species of trees.
data.loc[:,"confidence":] = utils.engineer._scaleData(data.loc[:,"confidence":])

X = np.array(data.loc[:, "confidence":])  # Number of clusters

contam = 0.05

# %%
# https://www.geeksforgeeks.org/novelty-detection-with-local-outlier-factor-lof-in-scikit-learn/
from sklearn.neighbors import LocalOutlierFactor
clf  = LocalOutlierFactor(n_neighbors=50)
outlier_scores  = clf.fit_predict( data.loc[:, 'confidence':])
outlierness = clf.negative_outlier_factor_

anomaly = data[outlier_scores == -1]
nominal = data[outlier_scores != -1]

plotA.plot(tryout, nominal, anomaly)

# %%
#     ABOD
from pyod.models.abod import ABOD

clf = ABOD(contamination=contam, n_neighbors=50)

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%
    # KPCA Reconstruction
from pyod.models.kpca import KPCA
clf = KPCA(contamination=0.05)

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%
from pyod.models.lunar import LUNAR
clf = LUNAR()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%

from pyod.models.mcd import MCD
clf = MCD()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%

from pyod.models.hbos import HBOS
clf = HBOS(n_bins="auto", contamination=contam)

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html#examples
# https://github.com/sahandha/eif/blob/master/Notebooks/TreeVisualization.ipynb 
# Set the predictors
h2o.init()
h2o_df = h2o.H2OFrame(data[list(data.columns)[4:]])
predictors = list(data.columns)[4:]
# https://github.com/sahandha/eif/blob/master/Notebooks/EIF.ipynb
    # Maybe this may help with plotting but I am uncertain

# Define an Extended Isolation forest model
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 1000,
                                          sample_size = int(data.shape[0] * 0.8),
                                          extension_level = 6)#len(predictors) - 1)
# Train Extended Isolation Forest
eif.train(x = predictors,
          training_frame = h2o_df)
# Calculate score
eif_result = eif.predict(h2o_df)
anomaly_score = eif_result["anomaly_score"]
mean_length = eif_result["mean_length"]

outlierness = eif_result.as_data_frame()

    # 0.5 is a good threshold, for a weak one go <= 0.4
    # for a tight one go >= 0.5 
anomaly = data[outlierness["anomaly_score"] > 0.48]
nominal = data[outlierness["anomaly_score"] <= 0.48]

plotA.plot(tryout, nominal, anomaly)

# %%
from  pyod.models.auto_encoder import AutoEncoder

clf = AutoEncoder()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %% 

from pyod.models.copod import COPOD
clf = COPOD(contamination=contam)

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)

# %%

from pyod.models.cblof import CBLOF

clf = CBLOF(contamination=contam)
X = np.array(data.loc[:, "confidence":])
clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly = data[outlier_scores == 1]
nominal = data[outlier_scores == 0]

plotA.plot(tryout, nominal, anomaly)





# %%

# TODO: Might be a nice visualise in the x and y direction for anomaly scoring
# 		https://plotly.com/python/3d-surface-plots/