# %%
import dataHandler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.plotting
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator

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
# https://github.com/sahandha/eif/blob/master/Notebooks/EIF.ipynb
    # Maybe this may help with plotting but I am uncertain
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
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
nominal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')

# %%



# %%
                    # Delauney Paper
# %%

# https://gis.stackexchange.com/questions/459091/definition-of-multipolygon-distance-in-shapely
import shapely.plotting

# shapely.plotting.plot_polygon(data.iloc[0,1], color = "red")
# shapely.plotting.plot_polygon(data.iloc[4,1], color = "blue")
# plt.show()

# # this distance outputs the distance from the nearest vertex to the nearest vertex of the
# # polygons not from the centroid to the centroid
# print("distance: ", {data.iloc[0,1].distance(data.iloc[4,1])})

# (-8.28502 - -8.28497) = 0.00005       # From closrset vertex to closest vertex
# (-8.28506 - -8.28495) = 0.00011       # From centroid to centroid\

# don't have to do polygons to polygons can do centroid to centtroid.

# %% 
                    # Delauney Paper


# %%

from libpysal import weights, examples
from libpysal.cg import voronoi_frames
from contextily import add_basemap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# read in example data from a geopackage file. Geopackages
# are a format for storing geographic data that is backed
# by sqlite. geopandas reads data relying on the fiona package,
# providing a high-level pandas-style interface to geographic data.
# Many different kinds of geographic data formats can be read by geopandas.
cases = data["centroid"]

# In order for networkx to plot the nodes of our graph correctly, we
# need to construct the array of coordinates for each point in our dataset.
# To get this as a numpy array, we extract the x and y coordinates from the
# geometry column.
coordinates = np.column_stack((cases.geometry.x, cases.geometry.y))

# While we could simply present the Delaunay graph directly, it is useful to
# visualize the Delaunay graph alongside the Voronoi diagram. This is because
# the two are intrinsically linked: the adjacency graph of the Voronoi diagram
# is the Delaunay graph for the set of generator points! Put simply, this means
# we can build the Voronoi diagram (relying on scipy.spatial for the underlying
# computations), and then convert these polygons quickly into the Delaunay graph.
# Be careful, though; our algorithm, by default, will clip the voronoi diagram to
# the bounding box of the point pattern. This is controlled by the "clip" argument.
cells, generators = voronoi_frames(coordinates, clip="convex hull")

# With the voronoi polygons, we can construct the adjacency graph between them using
# "Rook" contiguity. This represents voronoi cells as being adjacent if they share
# an edge/face. This is an analogue to the "von Neuman" neighborhood, or the 4 cardinal
# neighbors in a regular grid. The name comes from the directions a Rook piece can move
# on a chessboard.
delaunay = weights.Rook.from_dataframe(cells)

# Once the graph is built, we can convert the graphs to networkx objects using the
# relevant method.
delaunay_graph = delaunay.to_networkx()

# To plot with networkx, we need to merge the nodes back to
# their positions in order to plot in networkx
positions = dict(zip(delaunay_graph.nodes, coordinates))

# Now, we can plot with a nice basemap.
ax = cells.plot(facecolor="lightblue", alpha=0.50, edgecolor="cornsilk", linewidth=2)
try:  # Try-except for issues with timeout/parsing failures in CI
    add_basemap(ax)
except:
    pass

ax.axis("off")
nx.draw(
    delaunay_graph,
    positions,
    ax=ax,
    node_size=2,
    node_color="k",
    edge_color="k",
    alpha=0.8,
)
plt.show()
# %%
# Spatial proximity graph
from scipy.spatial import Delaunay
# need to give lat and long
#points = data.iloc[50:100,2:4].to_numpy()
points = np.array(data.loc[:, ["longitude", "latitude"]])
tri = Delaunay(points, incremental=True, qhull_options="Q14")

# %%
import igraph as ig

# %%

plt.figure(figsize=(25,25))
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()


# %%
# get mean edge length

from itertools import combinations
triangle = tri.simplices
all_edges = set([tuple(sorted(edge)) for item in triangle for edge in combinations(item,2)])
np.mean([np.linalg.norm(points[edge[0]]-points[edge[1]]) for edge in all_edges])

# %%
fig, ax = plt.subplots(figsize=(25, 25))
tryout.plot.imshow(ax=ax)
#ax.figure(figsize=(25,25))
ax.triplot(points[:,0], points[:,1], tri.simplices)
ax.plot(points[:,0], points[:,1], 'o')
#ax.show()
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



# %%
# For Correction purposes
    # False-positives
        # Remove observation
    # Over-segmentations   
        # TODO: https://medium.com/@jesse419419/understanding-iou-and-nms-by-a-j-dcebaad60652
        #       https://infoscience.epfl.ch/server/api/core/bitstreams/768fc3bc-f7d4-4533-825d-5c398995526d/content
    # Under-segmentations
        # https://arxiv.org/pdf/2202.08682
        # https://www.mdpi.com/2072-4292/12/5/767
    # False-negatives - https://www.mdpi.com/2072-4292/11/4/410
        # YOlOv10
        # Can train YOLO on all available orchards
        # following this we can use it as a trianed model.
        # we then isolate the area that has a potential false-negative and test using trained YOLOv10.

    # Can build probability map using - https://www.mdpi.com/2072-4292/12/5/767