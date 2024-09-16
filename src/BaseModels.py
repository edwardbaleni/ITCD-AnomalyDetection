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

# REfernce:
    # https://arxiv.org/pdf/2206.09426
    # https://pyod.readthedocs.io/en/latest/index.html

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

from pyod.models.iforest import IForest
clf = IForest()

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

	# Semi-supervised approach
	# So we need to do outlier detection to split anomalies from inliers
	# then we do novelty detection to see if an observation in the anomalies group is 
	# actually an anomaly
from pyod.models.copod import COPOD
from pyod.models.vae import VAE
from pyod.models.cof import COF

clf = COPOD()
X = np.array(data.loc[:, "confidence":])
clf.fit(X)
outlier_scores = clf.labels_
outlierness = clf.decision_scores_

anomaly_1 = data[outlier_scores == 1]
nominal_1 = data[outlier_scores == 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='blue')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='red')

# TODO: try a combination of semi-supervised approaches.
#		Reconstruction does not work. But neighbourhood approach does
#		Worth trying a couple
# 		Problem with this method is that it takes everything as not an outlier

lof = LocalOutlierFactor(novelty=True)
lof.fit(np.array(nominal_1.loc[:, "confidence":]))

	# Other method
# clf = COF()

# clf.fit(np.array(nominal_1.loc[:, "confidence":]))
# outlier_scores = clf.labels_
# outlierness = clf.decision_scores_

	# Doesn't work
# from sklearn.svm import OneClassSVM
# clf = OneClassSVM(gamma='auto').fit(nominal_1.loc[:, "confidence":])
# y_test_pred = clf.predict(anomaly_1.loc[:, "confidence":])

# %%
y_test_pred = lof.predict(anomaly_1.loc[:, "confidence":])

# # get the prediction on the test data
# y_test_pred = clf.predict(anomaly_1.loc[:, "confidence":])  # outlier labels (0 or 1)
# #y_test_scores = clf.decision_function(anomaly_1.loc[:, "confidence":])  # outlier scores

anomaly_2 = anomaly_1[y_test_pred == -1]
nominal_2 = anomaly_1[y_test_pred == 1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_2.plot(ax=ax, facecolor='none', edgecolor='red')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='white')
nominal_2.plot(ax=ax, facecolor='none', edgecolor='blue')





# %% 
# can run many at once

# https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py
# https://arxiv.org/pdf/2206.09426
# https://pyod.readthedocs.io/en/latest/index.html

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD
from pyod.models.cof import COF
from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.suod import SUOD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.kpca import KPCA
from pyod.models.lunar import LUNAR

# %%
# TODO: add neural networks, LOCI, SOS, COF, SOD

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
				 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
				 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
				 LOF(n_neighbors=50)]


# %%
random_state = 42

# Define the number of inliers and outliers
n_samples = 200
outliers_fraction = 0.1
clusters_separation = [0]

from numpy import percentile
import matplotlib.font_manager
import matplotlib.ticker as ticker

import umap

df_dr = data.loc[:, "confidence":]
#embedding = umap.UMAP(n_neighbors=5).fit_transform(np.array(df_dr))#X)
embedding = pd.DataFrame(umap.UMAP(n_neighbors=20).fit_transform(np.array(df_dr)))

# TODO: Lower dimensionality already achieved by UMAP, now we shou
# Plot points in 2D so we can use dimensionality reduction to do this
# 
random_state = 42

# Define the number of inliers and outliers
n_samples = 200
outliers_fraction = 0.1
clusters_separation = [0]

# Compare given detectors under given settings
# Initialize the data
import math
x_min = math.floor(min(embedding.iloc[:,0]))
x_max = round(max(embedding.iloc[:,0]))
y_min = math.floor(min(embedding.iloc[:,1]))
y_max = round(max(embedding.iloc[:,1]))

xx, yy = np.meshgrid(np.linspace(-1, x_max, 100), np.linspace(-1, y_max, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# Define nine outlier detection tools to be compared
classifiers = {
	'Angle-based Outlier Detector (ABOD)':
		ABOD(contamination=outliers_fraction),
	'K Nearest Neighbors (KNN)': KNN(
		contamination=outliers_fraction),
	'Average KNN': KNN(method='mean',
					   contamination=outliers_fraction),
	'Median KNN': KNN(method='median',
					  contamination=outliers_fraction),
	'Local Outlier Factor (LOF)':
		LOF(n_neighbors=35, contamination=outliers_fraction),

	'Isolation Forest': IForest(contamination=outliers_fraction,
								random_state=random_state),
	'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction,
									   random_state=random_state),
	'INNE': INNE(
		max_samples=2, contamination=outliers_fraction,
		random_state=random_state,
	),

	'Locally Selective Combination (LSCP)': LSCP(
		detector_list, contamination=outliers_fraction,
		random_state=random_state),
	'Feature Bagging':
		FeatureBagging(LOF(n_neighbors=35),
					   contamination=outliers_fraction,
					   random_state=random_state),
	'SUOD': SUOD(contamination=outliers_fraction),

	'Minimum Covariance Determinant (MCD)': MCD(
		contamination=outliers_fraction, random_state=random_state),

	# 'Principal Component Analysis (PCA)': PCA(
	# 	contamination=outliers_fraction, random_state=random_state),

	'Connectivity-Based Outlier Factor (COF)' : COF(contamination=outliers_fraction),

	'KPCA': KPCA(
		contamination=outliers_fraction),

	'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
												random_state=random_state),

	'LMDD': LMDD(contamination=outliers_fraction,
				 random_state=random_state),

	'Histogram-based Outlier Detection (HBOS)': HBOS(
		contamination=outliers_fraction),

	'Copula-base Outlier Detection (COPOD)': COPOD(
		contamination=outliers_fraction),

	'ECDF-baseD Outlier Detection (ECOD)': ECOD(
		contamination=outliers_fraction),
		
	'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

	'QMCD': QMCD(
		contamination=outliers_fraction),

	'Sampling': Sampling(
		contamination=outliers_fraction),

	'LUNAR': LUNAR(),

	'Variational Autoencoder (VAE)':
		VAE(contamination=outliers_fraction,
			  random_state=random_state),

	'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
}

# Show all detectors
for i, clf in enumerate(classifiers.keys()):
	print('Model', i + 1, clf)
	
# %%
from numpy import percentile
import matplotlib.font_manager
import matplotlib.ticker as ticker

# Fit the models with the generated data and
# compare model performances
for i, offset in enumerate(clusters_separation):
	np.random.seed(42)
	# Data generation
	X = np.array(data.loc[:, "confidence":])  

	# Fit the model
	plt.figure(figsize=(20, 22))
	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print()
		print(i + 1, 'fitting', clf_name)
		# fit the data and tag outliers
		clf.fit(X)
		outlier_scores = clf.labels_
		scores_pred = clf.decision_function(X) * -1
		y_pred = clf.predict(X)
		threshold = percentile(scores_pred, 100 * outliers_fraction)

		anomaly_1 = data[outlier_scores == 1]
		nominal_1 = data[outlier_scores == 0]
		ax = plt.subplot(5, 5, i + 1)
		tryout.plot.imshow(ax=ax)
		nominal_1.plot(ax=ax, facecolor = 'none',edgecolor='red') 
		anomaly_1.plot(ax=ax, facecolor = 'none',edgecolor='blue')
		ax.set_title("%d. %s" % (i + 1, clf_name))
		ax.axis("off")
plt.show()


# %%
# Fit the models with the generated data and
# compare model performances

for i, offset in enumerate(clusters_separation):
	np.random.seed(42)
	X = np.array(data.loc[:, "confidence":])
	X = np.array(embedding)  
	plt.figure(figsize=(22, 22)) # Fit the model
	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print()
		print(i + 1, 'fitting', clf_name)
		# fit the data and tag outliers
		clf.fit(X)
		scores_pred = clf.decision_function(X) * -1
		y_pred = clf.predict(X)
		threshold = percentile(scores_pred, 100 * outliers_fraction)
		
		# plot the levels lines and the points
		anomaly_1 = X[y_pred == 1]
		nominal_1 = X[y_pred == 0]
		
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
		Z = Z.reshape(xx.shape)
		print([threshold, Z.max()])
		n_outliers = (outlier_scores == 1)
		subplot = plt.subplot(5, 5, i + 1)
		subplot.contourf(xx, yy, Z, levels=np.linspace(min(Z.min(), threshold), max(Z.min(), threshold), 7),
						 cmap=plt.cm.Blues_r)
		# a = subplot.contour(xx, yy, Z, levels=[threshold],
		#                     linewidths=2, colors='red')
		subplot.contourf(xx, yy, Z, levels=sorted([threshold, Z.max()]),
						 colors='orange')
		b = subplot.scatter(nominal_1[:,0], nominal_1[:,1], c='white',
		 					s=20, edgecolor='k')
		c = subplot.scatter(anomaly_1[:,0], anomaly_1[:,1], c='black',
		 					s=20, edgecolor='k')
		#c = subplot.scatter(X[:, 0], X[:, 1], c='black',s=20, edgecolor='k')
		subplot.axis('tight')
		subplot.legend(
			[
				# a.collections[0],
				 b, c],
			[
				# 'learned decision function',
				'true inliers', 'true outliers'],
			prop=matplotlib.font_manager.FontProperties(size=10),
			loc='lower right')
		subplot.set_xlabel("%d. %s" % (i + 1, clf_name))
		subplot.set_xlim((-1, x_max))
		subplot.set_ylim((-1, y_max))
	plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
	plt.suptitle("25 outlier detection algorithms on synthetic data",
				 fontsize=35)
	
plt.show()



# %% 

# TODO: Ensemble
# 		https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.combination.majority_vote




# %%

# TODO: Might be a nice visualise in the x and y direction for anomaly scoring
# 		https://plotly.com/python/3d-surface-plots/