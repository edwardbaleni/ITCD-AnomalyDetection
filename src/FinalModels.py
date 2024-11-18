# %%
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.plotting
import networkx as nx
import utils.plotAnomaly as plotA

sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = utils.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
                              scale=True)
data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData
# For plotting
mask = mask
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255


# %%
# remove non-robust features - Doesn't help yet
# data = data[list(data.columns)[:5] + list(data.columns)[10:]]
# data.drop(['dist1', 'dist2', 'dist3', 'dist4'], axis = 1, inplace=True)



# %% 
                    # Delauney Paper

import utils.Triangulation as tri

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)

knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)



# %%
    # Plot Triangulations
tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)

tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)


# %%
# check if any nodes have no neighbours
# delaunay.islands

# %%
G = tri.setNodeAttributes(d_g, data)
G = tri.setEdgeAttributes(G, data)


# %%
# TODO: https://stackoverflow.com/questions/70452465/how-to-load-in-graph-from-networkx-into-pytorch-geometric-and-set-node-features
#       https://stackoverflow.com/questions/71011514/converting-a-pyg-graph-to-a-networkx-graph
import torch
from torch_geometric.utils.convert import from_networkx

pyg_graph = from_networkx(G, 
                          group_node_attrs = "all", 
                          group_edge_attrs= "distances")

# %%
# TODO: AD with just graph structure
# TODO: AD with graph structure plus attributes
    # TODO: https://docs.pygod.org/en/latest/tutorials/1_intro.html#sphx-glr-tutorials-1-intro-py
    # TODO: https://pytorch-geometric.readthedocs.io/en/latest/index.html

# train a dominant detector
from pygod.detector import CoLA

model = CoLA(gpu=0)  # hyperparameters can be set here # gpu = 0, uses gpu # gpu = 1 uses cpu
model.fit(pyg_graph)  # input data is a PyG data object

# get outlier scores on the training data (transductive setting)
label = model.label_
labels = label.detach().cpu().numpy()
score = model.decision_score_

anomaly = data[labels == 1]
nominal = data[labels == 0]

plotA.plot(tryout, nominal, anomaly)
# %%
# train a dominant detector
from pygod.detector import DMGD

model = DMGD()  # hyperparameters can be set here # gpu = 0, uses gpu # gpu = 1 uses cpu
model.fit(pyg_graph)  # input data is a PyG data object

# get outlier scores on the training data (transductive setting)
label = model.label_
labels = label.detach().cpu().numpy()
score = model.decision_score_

anomaly = data[labels == 1]
nominal = data[labels == 0]

plotA.plot(tryout, nominal, anomaly)


# %%
# TODO: performance metrics for outlier detection
#        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#sphx-glr-auto-examples-miscellaneous-plot-outlier-detection-bench-py


# %%

# local Geary C Statistic

import utils.Triangulation as tri
import networkx as nx

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)
knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)

# TODO: Only pass in necessary attributes
# TODO: do this in FinalModels as well

# tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)
# tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)

import esda
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import libpysal as lps
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

df = data
wq = knn_w #d_w#lps.weights.Rook.from_dataframe(df)
wq.transform = 'r'

# should compare knn weights to delauney weights
# but we already know that delauney is better!
w = d_w
# x1 = data["confidence"]
# x2 = data["NDVI_mean"]
# x3 = data["eccentricity"]
# x4 = data["roundness"]
# # x5 = data["z0"]
# # x6 = data["z1"]
# # x7 = data["z2"]
# # x8 = data["contrast"]
# # x9 = data["energy"]
# # x10 = data["bendingE"]
# xx = [x1,x2,x3,x4]#,x5,x6,x7,x10]
# lG_mv = esda.Geary_Local_MV(connectivity=w).fit(xx)

# # observed multivariate Local Geary values.
# lG_mv.localG[0:5] 
# # array containing the simulated p-values for each unit.
# # significance level of statistic
# lG_mv.p_sim[0:5]

# df = data
# f, ax = plt.subplots(1, figsize=(20, 20))
# tryout.plot.imshow(ax=ax)
# df.assign(cl= np.log10(lG_mv.localG)).plot(column='cl', categorical=False,
#         k=5, cmap='viridis', linewidth=0.1, ax=ax,
#         edgecolor='white', legend=True, alpha=0.7)
# ax.set_axis_off()
# plt.title("Geary C Multivariate Spatial Autocorrelation")

# plt.show()

# # p-value point
# f, ax = plt.subplots(1, figsize=(15, 15))
# tryout.plot.imshow(ax=ax)
# df.assign(cl= lG_mv.p_sim > 0.05).plot(column='cl', categorical=True,
#         k=5, cmap='viridis', linewidth=0.1, ax=ax,
#         edgecolor='black', legend=True, alpha=0.7)
# ax.set_axis_off()
# plt.title("Geary C Multivariate P-Value")

# plt.show()
# # observed multivariate Local Geary values. 

# anomaly = data[np.log(lG_mv.localG) >= 2.2]
# nominal = data[np.log(lG_mv.localG) < 2.2]

# plotA.plot(tryout, nominal, anomaly)







# %%


# but we already know that delauney is better!
w = d_w
xx = data.loc[:, "confidence":].values.T.tolist()
xx = [pd.Series(x) for x in xx]
lG_mv = esda.Geary_Local_MV(connectivity=w).fit(xx)

# observed multivariate Local Geary values.
lG_mv.localG[0:5] 
# array containing the simulated p-values for each unit.
# significance level of statistic
lG_mv.p_sim[0:5]

df = data
f, ax = plt.subplots(1, figsize=(20, 20))
tryout.plot.imshow(ax=ax)
df.assign(cl= np.log10(lG_mv.localG)).plot(column='cl', categorical=False,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='white', legend=True, alpha=0.7)
ax.set_axis_off()
plt.title("Geary C Multivariate Spatial Autocorrelation")

plt.show()
# observed multivariate Local Geary values. 

anomaly = data[np.log(lG_mv.localG) >= 0.5 * np.log(lG_mv.localG.max())]
nominal = data[np.log(lG_mv.localG) < 0.5 * np.log(lG_mv.localG.max())]

plotA.plot(tryout, nominal, anomaly)














# %%
# above method suffers from high dimensionality!
# import numpy as np

# from sklearn.decomposition import PCA
# X = np.array(data.loc[:, "confidence":]) 
# pca = PCA(n_components=10)

# pca.fit(X)
# PCA(n_components=10)

# plt.scatter(list(range(0,10)), pca.explained_variance_ratio_)

# import umap
# fit = umap.UMAP(n_components=10)
# u = fit.fit_transform(X)
# dat = pd.DataFrame(u)


# %%
# Grouping of anomalies into individual categories!

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