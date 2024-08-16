# %%

# How about we make this a whole file a class
# that works on one file at a time
# then we use another python file to loop over all images or folders
# and in this way this part can be parellizable
# since the algorhtims are each rely solely on the one image alone.


# %%
import shapely.plotting
import shapely
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
import pandas as pd
import math

import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator


# %%    
                    # Feature Selection (if too many features)

# TODO: Feature selection

# %%
# from heatmap import corrplot

# plt.figure(figsize=(20, 20))
# corrplot(a.corr())

import seaborn as sns
# ax = sns.heatmap(a.corr(), annot=True)
# g = sns.PairGrid(a.loc[:, "confidence":])
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# g.add_legend()

sns.heatmap(a.loc[:, "confidence":].corr(), annot=False, cmap="crest")




# %%

                    # Histogram Method
# %%
    # Can simply look into outliers in the data here
fig = plt.figure(figsize =(10, 10))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(a.loc[:,"confidence":])
# show plot
plt.show()



# %%

                    # Extended Isolation Forest

# %%
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html#examples
# https://github.com/sahandha/eif/blob/master/Notebooks/TreeVisualization.ipynb 
# Set the predictors
h2o.init()
# %%
h2o_df = h2o.H2OFrame(a.loc[:,[ 'confidence',
       'crown_projection_area', 'crown_perimeter', 'radius_of_gyration',
       #'short', 'long', 
       'minor_axis', 'major_axis', 
       'isoperimetric', 'shape_index', 'form_factor', 'circularity',
       'convexity', 'solidity', 'elongation', 'roundness', 'DEM_max', 'NDRE_max',
       'NDVI_max', 'GNVDI_max', 'ENDVI_max', #'NDRE_median', 
       #'NDVI_median',
       #'GNVDI_median', 'ENDVI_median', 
       'dist1', 'dist2', 'dist3', 'dist4']])
predictors = [ 'confidence',
       'crown_projection_area', 'crown_perimeter', 'radius_of_gyration',
       #'short', 'long', 
       'minor_axis', 'major_axis', 
       'isoperimetric', 'shape_index', 'form_factor', 'circularity',
       'convexity', 'solidity', 'elongation', 'roundness', 'DEM_max', 'NDRE_max',
       'NDVI_max', 'GNVDI_max', 'ENDVI_max', #'NDRE_median', 
       #'NDVI_median',
       #'GNVDI_median', 'ENDVI_median', 
       'dist1', 'dist2', 'dist3', 'dist4']#list(a.columns)

# %%
# Extended Isolation Forest is a great unsupervised method for anomaly detection
# however, it does not allow for the use of spatial features

# Define an Extended Isolation forest model
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 1000,
                                          sample_size = int(len(a) * 0.8),
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
# for delineations[1]
# when the confidence variable is included then use a thresholf of 0.65
# when it is not included then use a threshold of 0.5, however, this picks out a lot more anomalies
# that may in fact be normal.
# It is likely that most papers will not include the confidence variable

# when scaled and no confidence then 0.5 is a good threshold, most of the anomalies are captured
# the same anomalies are captured with or withour using confidence anyway. So we may be able to get away with
# not using confidence
anomaly = a[b["anomaly_score"] > 0.5]
nominal = a[b["anomaly_score"] <= 0.5]

# %%

# fig, ax = plt.subplots(figsize=(20, 20))
# rio.plot.show(clipped, ax=ax)
# anomaly.plot(ax=ax, facecolor='none', edgecolor='red')
# nominal.plot(ax=ax, facecolor='none', edgecolor='blue')

#mask = gpd.read_file("C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/122075/survey_polygon.geojson")
#tryout = xds_RGB[0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
#tryout = tryout/255
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly.plot(ax=ax, facecolor='none', edgecolor='red')
nominal.plot(ax=ax, facecolor='none', edgecolor='blue')
# nominal.iloc[1:].plot(ax=ax, facecolor='none', edgecolor='blue')
# nominal.iloc[0:1].plot(ax=ax, facecolor='none', edgecolor='orange')






# %%
# https://www.geeksforgeeks.org/ml-fuzzy-clustering/
    # fuzzy means does not work
import skfuzzy as fuzz
from skfuzzy import control as ctrl

X = np.array(a.loc[:, "confidence":])  # Number of clusters

# Define the number of clusters
n_clusters = 3
 
# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)
 
# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)
 
# # Print the cluster centers
# print('Cluster Centers:', cntr)
 
# # Print the cluster membership for each data point
# print('Cluster Membership:', cluster_membership)

n1 = a[cluster_membership == 0]
n2 = a[cluster_membership == 1] 
n3 = a[cluster_membership == 2]
# n4 = a[cluster_membership == 3]
# n5 = a[cluster_membership == 4]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
n1.plot(ax=ax, facecolor='none', edgecolor='red')
n2.plot(ax=ax, facecolor='none', edgecolor='blue')
n3.plot(ax=ax, facecolor='none', edgecolor='purple')
# n4.plot(ax=ax, facecolor='none', edgecolor='green')


# %%
# https://www.geeksforgeeks.org/novelty-detection-with-local-outlier-factor-lof-in-scikit-learn/
from sklearn.neighbors import LocalOutlierFactor
# TODO: this is not the correct way to do this,
#       look into https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-novelty-detection-py

lof_outlier  = LocalOutlierFactor(n_neighbors=20)
outlier_scores  = lof_outlier.fit_predict( a.loc[:, 'confidence':])

anomaly_1 = a[outlier_scores == -1]
nominal_1 = a[outlier_scores != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='red')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='blue')

# # %%
#     # Robust covariance
#             # https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#sphx-glr-auto-examples-covariance-plot-mahalanobis-distances-py
# from sklearn.covariance import EmpiricalCovariance, MinCovDet
# X = a.loc[:, 'confidence':]
# # fit a MCD robust estimator to data
# robust_cov = MinCovDet().fit(X)
# # fit a MLE estimator to data
# emp_cov = EmpiricalCovariance().fit(X)
# print(
#     "Estimated covariance matrix:\nMCD (Robust):\n{}\nMLE:\n{}".format(
#         robust_cov.covariance_, emp_cov.covariance_
#     )
# )


# %%
    # Unsupervised SVM
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
from sklearn.svm import OneClassSVM
X = a.loc[:, 'confidence':]
clf = OneClassSVM(gamma='auto').fit(X)
scores = clf.predict(X)
#clf.score_samples(X)
anomaly_1 = a[scores == -1]
nominal_1 = a[scores != -1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='red')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='blue')


# %%

# Above methods are in line with: https://ieeexplore-ieee-org.ezproxy.uct.ac.za/document/9297055
# Paper says that isolation forests are the best option for AD



# %%



            # do not do delauney triangulation with EFI use, nearest neighbour instead to get closest points 
            # otherwise the number of variables won't be contained at each vairable
            # Only use delauney triangulation for second method
            # Use distatnces to centres not to vertices

# %%
                    # Delauney Paper
# %%

# https://gis.stackexchange.com/questions/459091/definition-of-multipolygon-distance-in-shapely
import shapely.plotting

shapely.plotting.plot_polygon(a.iloc[0,1], color = "red")
shapely.plotting.plot_polygon(a.iloc[4,1], color = "blue")
plt.show()

# this distance outputs the distance from the nearest vertex to the nearest vertex of the
# polygons not from the centroid to the centroid
print("distance: ", {a.iloc[0,1].distance(a.iloc[4,1])})

# (-8.28502 - -8.28497) = 0.00005       # From closrset vertex to closest vertex
# (-8.28506 - -8.28495) = 0.00011       # From centroid to centroid\

# don't have to do polygons to polygons can do centroid to centtroid.

# %% 
from scipy.spatial import Delaunay
points = a.iloc[0:50,5:7].to_numpy()
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