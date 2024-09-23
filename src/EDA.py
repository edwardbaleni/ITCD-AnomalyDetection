# %%
import dataHandler

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import shapely

import seaborn as sns
import plotly.express as px

# TODO: Speed up dataCollect

# working directory is that where the file is placed
# os.chdir("..")
sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = dataHandler.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 1

# start = timer()
myData = dataHandler.engineer(num, data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped, 
                              scale=False)
# end = timer()
# print(end - start)

data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData

# Plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, 
                                           mask.crs, 
                                           drop=True, 
                                           invert=False)
tryout = tryout/255

# %%

    # TODO: For the spectral indices and vegetative indices we
    #       just need to do profile plots to understand relevance
    #       Can also look at local spatial autocorrelations
    #       Can also view as spatial points and quickly perform a spatial points analysis
    #       Just to gain as much of an understanding on the spectral data as possible
    #       Can do the same for the morphological properties just to better understand 
    #       what makes an anomaly an anomaly.

    # TODO: Can do an EDA on graph infrastructure
    #       https://towardsdatascience.com/eda-on-graphs-via-networkx-a79d2684da53
    
    # TODO: What we can do for the EDA is look at morphological and image properties separately
    #       Then look at the most significant of these together


# %%

pd.DataFrame(data.loc[:,"confidence":]).plot()
plt.show()


# %%


g = sns.PairGrid(data.loc[:,"confidence":], diag_sharey=False, corner=True)
g.map_lower(plt.scatter, alpha = 0.6)
g.map_diag(plt.hist, alpha = 0.7)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot)


# %%
fig = px.scatter_matrix(data.loc[:,"confidence":])
fig.show()

# %%

pd.plotting.scatter_matrix(data.loc[:,"confidence":], alpha=0.2)

# %% 
# plot data

#plt.scatter(data["confidence"], data["dist1"]) 
#plt.scatter(data["confidence"], data["NDVI_mean"])
fig = px.scatter(x = data["confidence"], y = data["elongation"] )
fig.add_scatter(x = data["confidence"], y = data["NDVI_mean"])
fig.show()

# %%

fig = px.scatter(data, 
                 x = "latitude",
                 y = "longitude", 
                 size = "confidence", 
                 size_max=5,
                 hover_data=["dist1", "NDVI_mean", "elongation"])
#data.loc[:, "crown_projection_area":].columns

fig.show()

# %%
# import plotly.graph_objects as go
# tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
# tryout = tryout/255
# fig = go.Figure(go.Image(z=np.array(tryout)))
# fig.show()

# %%



fig = px.imshow(np.array(data.loc[:,"confidence":].corr()), text_auto=True, aspect=True,
                x = list(data.loc[:,"confidence":].columns),
                y = list(data.loc[:,"confidence":].columns))
fig.show()
# %%    
                    # Feature Selection (if too many features)



# TODO: Feature selection
#      - tSNE
#      - IsoMap
#      - feature clustering
#      - UMAP



# %%

from statsmodels.stats.outliers_influence import variance_inflation_factor

# the independent variables set
X = data.loc[:,"confidence":]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns


# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

print(vif_data)

# %%

# calculate correlation values
correlation_matrix = data.loc[:,"confidence":].corr()
correlation_matrix

# %%

import seaborn as sns

sns.clustermap(data.loc[:, "confidence":].corr(), annot=True)

# %%

from sklearn.feature_selection import VarianceThreshold

X = data.loc[:,"confidence":]

selector = VarianceThreshold(threshold=1)

selector.fit_transform(X)

# outputting low variance columns
concol = [column for column in data.loc[:,"confidence":].columns 
          if column not in data.loc[:,"confidence":].columns[selector.get_support()]]

for features in concol:
    print(features)

# drop low variance columns
X.drop(concol, axis = 1)


# %%
# https://doi.org/10.1016/j.jocs.2021.101502
# https://stats.stackexchange.com/questions/108743/methods-in-r-or-python-to-perform-feature-selection-in-unsupervised-learning
# Gives exact oppositie to VariableThresholding
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]


import numpy as np
#X = np.random.random((1000,1000))
X = data.loc[:,"confidence":]
pfa = PFA(n_features=15)
pfa.fit(np.array(X))

# To get the transformed matrix
Y = pfa.features_

# To get the column indices of the kept features
column_indices = pfa.indices_

print(data.loc[:,"confidence":].iloc[:, column_indices])



# %%

# TODO: https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/SPEC.py

# TODO: https://doi.org/10.1016/j.chemolab.2021.104396
#       https://www.google.com/search?client=firefox-b-d&q=filter+and+hybrid+filter-wrapper+feature+subset+selection

# TODO: https://www.jmlr.org/papers/volume5/dy04a/dy04a.pdf
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

                    # Histogram          
    # Can simply look into outliers in the data here
fig = plt.figure(figsize =(10, 10))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(data.loc[:,"confidence":])
# show plot
plt.show()



# %%
# TODO: spatial EDA

import utils.Triangulation as tri
import networkx as nx

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)
knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)

# TODO: Only pass in necessary attributes
# TODO: do this in FinalModels as well

# graph plots well even with attributes added in
G_delauney = tri.setNodeAttributes(d_g, data)
G_delauney = tri.setEdgeAttributes(G_delauney, data)

# tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)
# tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)

# %%

# %%
import esda
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import libpysal as lps
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

df = data
wq = knn_w#d_w#lps.weights.Rook.from_dataframe(df)
wq.transform = 'r'

y = df['roundness']
ylag = lps.weights.lag_spatial(wq, y)

import mapclassify as mc
ylagq5 = mc.Quantiles(ylag, k=5)

f, ax = plt.subplots(1, figsize=(9, 9))
df.assign(cl=ylagq5.yb).plot(column='cl', categorical=True, \
        k=5, cmap='GnBu', linewidth=0.1, ax=ax, \
        edgecolor='white', legend=True)
ax.set_axis_off()
plt.title("Spatial Lag Median Price (Quintiles)")

plt.show()

# %%
mi = esda.moran.Moran(y, wq)
mi.I
# here we basically observe no spatial autocorrelation
import seaborn as sbn
sbn.kdeplot(mi.sim, shade=True)
plt.vlines(mi.I, 0, 1, color='r')
plt.vlines(mi.EI, 0,1)
plt.xlabel("Moran's I")

# %%

# single variable local morans I
li = esda.moran.Moran_Local(y, wq)

df = data
f, ax = plt.subplots(1, figsize=(20, 20))
tryout.plot.imshow(ax=ax)
df.assign(cl= li.Is).plot(column='cl', categorical=False,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='white', legend=True, alpha = 0.7)
ax.set_axis_off()
plt.title("LISA Spatial Autocorrelation")

plt.show()

# %%
# TODO: https://www-jstor-org.ezproxy.uct.ac.za/stable/2684298?sid=primo&seq=6

# %%

# TODO: Go through networkx and see if there are any more 
#       Network analysis tools that would be helpful in EDA

# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality
print(nx.degree_centrality(G_delauney))

# TODO: Assortivity as part of an analysis, used to understand local structure of networks
    # https://doi.org/10.1038/s41598-020-78336-9
        # https://networkx.org/documentation/stable/reference/algorithms/assortativity.html 
print(nx.numeric_assortativity_coefficient(G_delauney, "confidence"))
print(nx.degree_pearson_correlation_coefficient(G_delauney))
# TODO: Read more into this one
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_neighbor_degree.html#networkx.algorithms.assortativity.average_neighbor_degree
print(nx.average_neighbor_degree(G_delauney))#, weight="weight")


# %%

# TODO: Analyse the structure of our graphs
    # https://www.nature.com/articles/s41598-020-69795-1?fromPaywallRec=false
    # https://www.nature.com/articles/srep31708 - code below can also be used to find sub-graphs (neighbourhoods)
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.onion_layers.html#networkx.algorithms.core.onion_layers

# %%

# TODO: ESDA on at least one variable of each category





# %%
# Multivariate Spatial Autocorrelation
# https://onlinelibrary.wiley.com/doi/epdf/10.1111/gean.12164
# https://www.jstor.org/stable/143141?origin=crossref
    # suffers from the curse of dimensionality so pick features wisely. 
# 

w = d_w
x1 = data["confidence"]
x2 = data["NDVI_mean"]
x3 = data["elongation"]
x4 = data["roundness"]

lG_mv = esda.Geary_Local_MV(connectivity=w).fit([x1,x2,x3,x4])

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

# p-value point
f, ax = plt.subplots(1, figsize=(15, 15))
tryout.plot.imshow(ax=ax)
df.assign(cl= lG_mv.p_sim > 0.05).plot(column='cl', categorical=True,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='black', legend=True, alpha=0.7)
ax.set_axis_off()
plt.title("Geary C Multivariate P-Value")

plt.show()


# %%
# Variogram clouds and spatial autocorrelation and https://www.tandfonline.com/doi/abs/10.1080/10618600.1999.10474812 
# are interesting tools for detecting outliers in a univariate setting

# TODO: https://link.springer.com/article/10.1007/s00362-013-0524-z#Sec2
#       This paper is great for exploration of multivariate spatial data
#       https://cran.r-project.org/web/packages/mvoutlier/index.html\

# from rpy2.robjects.packages import importr
# from rpy2.robjects import r, pandas2ri
# import rpy2.robjects as ro

# # rprint = ro.globalenv.get("print")
# geometry = data.loc[:, "centroid":]
# geometry.rename(columns={"centroid" : "geometry"}, inplace=True)
# geometry.set_crs(data.crs, inplace=True)
# geometry.to_crs(3857, inplace=True)

# df = geometry.loc[:,"confidence":].copy()
# df = pd.DataFrame(dataHandler.engineer._scaleData(df), 
#                   columns = list(df.columns))

# with (ro.default_converter + pandas2ri.converter).context():
#   r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)

# val = 1
# geometry["latitude"] = geometry["geometry"].x + 6.43*10**6
# geometry["longitude"] = geometry["geometry"].y + 3.644*10**6
# la = ro.IntVector( round(geometry["latitude"] * val) )
# lo = ro.IntVector( (geometry["longitude"] * val) )

# ylim = ro.IntVector([ np.array( geometry["latitude"]).min() * val , np.array(geometry["latitude"]).max() * val  ])

# mvoutlier = importr("mvoutlier")

# # dat, X, Y

# grdevices = importr('grDevices')

# grdevices.png(file="file.png", width=1000, height=1000)
# mvoutlier.locoutSort(dat=r_from_pd_df, 
#                      X=lo, 
#                      Y=la, 
#                      ylim = ylim)
# # plotting code here
# grdevices.dev_off()


# %% 
# TODO: Obtain Texture features
#       https://medium.com/@giakoumoglou4/pyfeats-open-source-software-for-image-feature-extraction-47f43bb33563
#       https://github.com/giakoumoglou/pyfeats/tree/main
#       https://pyradiomics.readthedocs.io/en/latest/usage.html#voxel-based-extraction
#       https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/?source=post_page-----cb1feb2dbd73--------------------------------
#       

# import cv2 as cv
#     # have to transpose the image a couple of times to get it right.
#     # but here we have a grayscale image
# im = cv.cvtColor(spectralData["rgb"].T.to_numpy()[:,:,:3], cv.COLOR_BGR2GRAY).T

# #     # get the masks
# # a = spectralData["rgb"].rio.clip(delineations)
# # ma = cv.cvtColor(a.T.to_numpy()[:,:,:3], cv.COLOR_BGR2GRAY).T


# # %%
# import pyfeats

# features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(im, ignore_zeros=True)

# # features, labels = pyfeats.fos(im, ma)


# # %%
# # https://www.kaggle.com/code/datascientistsohail/texture-features-extraction
# import skimage
# # this get the first geometry
# # we can use this to get texture properties
# touch = spectralData["rgb"].rio.clip([delineations.iloc[0,0]], spectralData["rgb"].rio.crs)
# im = cv.cvtColor(touch.T.to_numpy()[:,:,:3], cv.COLOR_BGR2GRAY).T

# # co_matrix = skimage.feature.graycomatrix(im, [5], [0], levels=256, symmetric=True, normed=True)

# # # Calculate texture features from the co-occurrence matrix
# # # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
# # contrast = skimage.feature.graycoprops(co_matrix, 'contrast')
# # correlation = skimage.feature.graycoprops(co_matrix, 'correlation')
# # energy = skimage.feature.graycoprops(co_matrix, 'energy')
# # homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')

# # # Print the texture features
# # print("Contrast:", contrast)
# # print("Correlation:", correlation)
# # print("Energy:", energy)
# # print("Homogeneity:", homogeneity)



# features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(im, ignore_zeros=True)

# # a["NDRE_median"] = a["geometry"].progress_apply(lambda x: float(data["NDRE"].rio.clip( [x], data["NDRE"].rio.crs).median()))

# # touch.plot()
