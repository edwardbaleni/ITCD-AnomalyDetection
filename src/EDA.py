# %%
import dataHandler

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import shapely

# TODO: Speed up dataCollect

# working directory is that where the file is placed
# os.chdir("..")
sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = dataHandler.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0

# start = timer()
myData = dataHandler.engineer(num, data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
# end = timer()
# print(end - start)

data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData

# # Plotting
# mask = mask
# tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
# tryout = tryout/255
# fig, ax = plt.subplots(figsize=(15, 15))
# tryout.plot.imshow(ax=ax)
# delineations.plot(ax=ax, facecolor = 'none',edgecolor='red') 


# %%    
                    # Feature Selection (if too many features)

# TODO: Feature selection
#      - tSNE
#      - IsoMap
#      - feature clustering
#      - UMAP

# %%
# To help with feature selection
from sklearn.manifold import TSNE
import plotly.express as px

df_dr = data.loc[:, "confidence":]
df_dr = df_dr.T

    # visualise
TSNE_model = TSNE(n_components=2, perplexity=3)
df_tsne = pd.DataFrame(TSNE_model.fit_transform(np.array(df_dr)))

df_tsne['entity'] = df_dr.index
df_tsne["theme"] = df_tsne["entity"].apply(lambda d : d[0:4])

fig_tsne = px.scatter(data_frame=df_tsne, x=0, y=1, hover_name='entity', color = "theme",title='T-SNE With 2 Components',)
fig_tsne.show()

# %%

import umap

df_dr = data.loc[:, "confidence":]
df_dr = df_dr.T

#embedding = umap.UMAP(n_neighbors=5).fit_transform(np.array(df_dr))#X)
df_umap = embedding = pd.DataFrame(umap.UMAP(n_neighbors=5).fit_transform(np.array(df_dr)))#X)
df_umap['entity'] = df_dr.index
df_umap["theme"] = df_umap["entity"].apply(lambda d : d[0:4])

fig_umap = px.scatter(data_frame=df_umap, x=0, y=1, hover_name='entity', color = "theme",title='T-SNE With 2 Components',)
fig_umap.show()


# %%
    #               Variable clustering
    #               Can pick one variable from each cluster
from varclushi import VarClusHi
def varClust(X):
    demo1_vc = VarClusHi(X,maxeigval2=1,maxclus=None)
    demo1_vc.varclus()
    demo1_vc.info
    demo1_vc.rsquare

    data = []
    for i in demo1_vc.rsquare["Cluster"].unique():
        check = demo1_vc.rsquare[demo1_vc.rsquare["Cluster"] == i ]
        data.append(check.iloc[0,1])

    X = X.loc[:,data]

    return(list(X.columns), demo1_vc)

variables, demo1 = varClust(data.loc[:, "confidence":])
print(variables)
print(demo1.rsquare)

# %%

import seaborn as sns

sns.heatmap(data.loc[:, "confidence":].corr(), annot=True, cmap="crest")




# %%

                    # Histogram Method
                    
    # Can simply look into outliers in the data here
fig = plt.figure(figsize =(10, 10))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(data.loc[:,"confidence":])
# show plot
plt.show()
# %%


# %%

# How about we make this a whole file a class
# that works on one file at a time
# then we use another python file to loop over all images or folders
# and in this way this part can be parellizable
# since the algorhtims are each rely solely on the one image alone.


# %%
import shapely.plotting
# import geopandas as gpd
# import rasterio as rio
# from rasterio.plot import show
# from osgeo import ogr, gdal
# from osgeo import gdalconst
# from rasterio.mask import mask
# import earthpy.spatial as es
# import earthpy.plot as ep
# import earthpy as et
# import matplotlib.pyplot as plt
# import numpy as np

import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator

from sklearn.cluster import HDBSCAN


# remove non-robust features - Doesn't help yet
# data = data[list(data.columns)[:5] + list(data.columns)[10:]]
# %%

                    # Extended Isolation Forest

# %%
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html#examples
# https://github.com/sahandha/eif/blob/master/Notebooks/TreeVisualization.ipynb 
# Set the predictors
h2o.init()
h2o_df = h2o.H2OFrame(data[list(data.columns)[4:]])
predictors = list(data.columns)[4:]

# %%
# Extended Isolation Forest is a great unsupervised method for anomaly detection
# however, it does not allow for the use of spatial features

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

# Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
# or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
anomaly_score = eif_result["anomaly_score"]

# Average path length  of the point in Isolation Trees from root to the leaf
mean_length = eif_result["mean_length"]

# %%
b = eif_result.as_data_frame()

    # 0.5 is a good threshold, for a weak one go <= 0.4
    # for a tight one go >= 0.5 
anomaly = data[b["anomaly_score"] > 0.4]
nominal = data[b["anomaly_score"] <= 0.4]

# %%


# Plotting
mask = mask
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
nominal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')


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
# can run many at once



# %%
                    # Delauney Paper
# %%

# https://gis.stackexchange.com/questions/459091/definition-of-multipolygon-distance-in-shapely
import shapely.plotting

shapely.plotting.plot_polygon(data.iloc[0,1], color = "red")
shapely.plotting.plot_polygon(data.iloc[4,1], color = "blue")
plt.show()

# this distance outputs the distance from the nearest vertex to the nearest vertex of the
# polygons not from the centroid to the centroid
print("distance: ", {a.iloc[0,1].distance(a.iloc[4,1])})

# (-8.28502 - -8.28497) = 0.00005       # From closrset vertex to closest vertex
# (-8.28506 - -8.28495) = 0.00011       # From centroid to centroid\

# don't have to do polygons to polygons can do centroid to centtroid.

# %% 
from scipy.spatial import Delaunay
points = data.iloc[0:50,5:7].to_numpy()
tri = Delaunay(points)


# %%
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()

# try both delauney triangulation method and a nearest neighbour or could use a sort

# %%

# fig, ax = plt.subplots(figsize=(20, 20))
# rio.plot.show(clipped, ax=ax)
# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()
          


# %%

# because distances aren't already worked out from Delauney, we can
# do this manually from polygon to polygon instead of from vertex to vertex

# Simplices are the indices of the vertices that make up the triangle in
# points. If we match this to the centroid in main dataframe then we could 
# find distances between polygons.

# TODO: performance metrics for outlier detection
#        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#sphx-glr-auto-examples-miscellaneous-plot-outlier-detection-bench-py