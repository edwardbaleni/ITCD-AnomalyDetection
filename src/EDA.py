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

data = myData.data.copy(deep=True)
delineations = myData.delineations.copy(deep=True)
mask = myData.mask.copy(deep=True)
spectralData = myData.spectralData
erf_num = myData.erf

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

shape = data.loc[:, "crown_projection_area":"bendingE"]
dist = data.loc[:, "dist1":"dist4"]
spec = data.loc[:, "DEM_mean":]

from pypalettes import get_hex
palette = get_hex("VanGogh3", keep_first_n=8)


# %%
# Note that the palette is set as a global variable can change this later.
def boxplot(dat, lo = True):
    sns.set_theme(style="darkgrid")
    Props = {'boxprops':{"alpha":0.7, 'edgecolor':palette[3], 'facecolor':palette[2]},
            'medianprops':{'color':palette[3]},
            'whiskerprops':{'color':palette[3]},
            'capprops':{'color':palette[3]},
            'flierprops':{'color':palette[3]}
            }

    fig, ax = plt.subplots(figsize=(15, 10))
    if (lo):
        ax.set(yscale="log")
    sns.boxplot(data=dat, 
                ax=ax, 
                linewidth=2,
                #color=palette[0], 
                **Props)

    ax.tick_params("x", labelrotation=45)
    

boxplot(spec)
boxplot(shape.iloc[:,:-1])
boxplot(shape[["bendingE"]])
boxplot(dist, False)

# %%


g = sns.PairGrid(shape, diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
g.map_diag(plt.hist, alpha = 1, bins=30, color = palette[3])
g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

g = sns.PairGrid(dist, diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
g.map_diag(plt.hist, alpha = 1, bins=30,color = palette[3])
g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

g = sns.PairGrid(spec, diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
g.map_diag(plt.hist, alpha = 1, bins=30,color = palette[3])
g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

# %%    
                    # Feature Selection

# %%
    # calculate correlation values
    # Recognise Multicollinearities
sns.clustermap(spec.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")# palette)
sns.clustermap(dist.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")
sns.clustermap(shape.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")

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

# TODO: Exhaustive search of unsupervised feature selection techniques
        # https://medium.com/analytics-vidhya/feature-selection-extended-overview-b58f1d524c1c





# %%
# TODO: spatial EDA

import utils.Triangulation as tri
import networkx as nx

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)
knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)

# TODO: Only pass in necessary attributes
# TODO: do this in FinalModels as well

# tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)
# tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)

# %%

# TODO: ESDA on at least one variable of each category
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
# Plasma is also a good colour
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
c = esda.Geary(y, wq)
c.C
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
w = d_w
xx = y
ci = esda.Geary_Local(connectivity=w).fit(xx)

df = data
f, ax = plt.subplots(1, figsize=(20, 20))
tryout.plot.imshow(ax=ax)
df.assign(cl= np.array(ci.localG)).plot(column='cl', categorical=False,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='white', legend=True, alpha = 0.7)
ax.set_axis_off()
plt.title("Geary Local Spatial Autocorrelation")

plt.show()

# %%
# Multivariate Spatial Autocorrelation
# https://onlinelibrary.wiley.com/doi/epdf/10.1111/gean.12164
# https://www.jstor.org/stable/143141?origin=crossref
    # suffers from the curse of dimensionality so pick features wisely. 
# 
data.loc[:,"confidence":] = dataHandler.engineer._scaleData(data.loc[:,"confidence":])

w = d_w
x1 = data["confidence"]
x2 = data["NDVI_mean"]
x3 = data["elongation"]
x4 = data["roundness"]
# x5 = data["z0"]
# x6 = data["z1"]
# x7 = data["z2"]
# x8 = data["contrast"]
# x9 = data["energy"]
# x10 = data["bendingE"]
xx = [x1,x2,x3,x4]#,x5,x6,x7,x10]
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

# TODO: Go through networkx and see if there are any more 
#       Network analysis tools that would be helpful in EDA

# graph plots well even with attributes added in
G_delauney = tri.setEdgeAttributes(tri.setNodeAttributes(d_g, data), 
                                   data)

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

