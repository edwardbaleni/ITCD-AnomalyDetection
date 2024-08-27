# %%
import dataHandler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = dataHandler.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = dataHandler.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped)
data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData
# For plotting
mask = mask
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255

# %%

# Below methods are in line with: https://ieeexplore-ieee-org.ezproxy.uct.ac.za/document/9297055
# Paper says that isolation forests are the best option for AD



# %%
# DBSCAN
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
# https://dinhanhthi.com/note/dbscan-hdbscan-clustering/
X = np.array(data.loc[:, "confidence":])  # Number of clusters

hdb = HDBSCAN(min_cluster_size=5)
hdb.fit(X)
hdb.labels_

anomaly_1 = data[hdb.labels_ <= 0] # data[hdb.labels_ == -1]
nominal_1 = data[hdb.labels_ > 0] # data[hdb.labels_ != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# %%
from sklearn.cluster import DBSCAN

import numpy as np

clustering = DBSCAN(eps=3, min_samples=10).fit(X)

clustering.labels_

anomaly_1 = data[hdb.labels_ < 0] # data[hdb.labels_ == -1]
nominal_1 = data[hdb.labels_ >= 0] # data[hdb.labels_ != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')


# %%
# https://www.geeksforgeeks.org/novelty-detection-with-local-outlier-factor-lof-in-scikit-learn/
from sklearn.neighbors import LocalOutlierFactor
lof_outlier  = LocalOutlierFactor(n_neighbors=50)
outlier_scores  = lof_outlier.fit_predict( data.loc[:, 'confidence':])

anomaly_1 = data[outlier_scores == -1]
nominal_1 = data[outlier_scores != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')


# %%
    # Elliptic Envelope
from sklearn.covariance import EllipticEnvelope
X = np.array(data.loc[:, "confidence":])  
outlier_scores = EllipticEnvelope(random_state=0).fit_predict(X)

anomaly_1 = data[outlier_scores == -1]
nominal_1 = data[outlier_scores != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# %%
#     ABOD
from pyod.models.abod import ABOD

clf = ABOD()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')
# %%
    # KPCA Reconstruction
from pyod.models.kpca import KPCA
clf = KPCA()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')


# %%
from pyod.models.lunar import LUNAR
clf = LUNAR()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')


# %%

from pyod.models.mcd import MCD
clf = MCD()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')
# %%

from pyod.models.hbos import HBOS
clf = HBOS(n_bins="auto")

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# %% 

from pyod.models.vae import VAE

clf = VAE()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# %% 

from pyod.models.dif import DIF

clf = DIF()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# %%
from  pyod.models.auto_encoder import AutoEncoder

clf = AutoEncoder()

clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')


# %% 
# can run many at once