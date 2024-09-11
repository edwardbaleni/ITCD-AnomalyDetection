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
num = 0

# start = timer()
myData = dataHandler.engineer(num, data_paths_tif, data_paths_geojson, data_paths_geojson_zipped, scale=False)
# end = timer()
# print(end - start)

data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData

# Plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
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
px.imshow(tryout)
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
    # TODO: https://www.theoj.org/joss-papers/joss.02869/10.21105.joss.02869.pdf
    # TODO: https://networkx.org/documentation/stable/auto_examples/geospatial/extended_description.html
    # TODO: https://pysal.org/notebooks/explore/esda/intro.html
            # Explore this pysal library for EDA stuff

import utils.Triangulation as tri
import networkx as nx

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)
knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)

tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)
tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)

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
wq = knn_w#lps.weights.Rook.from_dataframe(df)
wq.transform = 'r'

y = df['confidence']
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
# TODO: https://www-jstor-org.ezproxy.uct.ac.za/stable/2684298?sid=primo&seq=6