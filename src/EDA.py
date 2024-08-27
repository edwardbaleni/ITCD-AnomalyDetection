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

                    # Histogram          
    # Can simply look into outliers in the data here
fig = plt.figure(figsize =(10, 10))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(data.loc[:,"confidence":])
# show plot
plt.show()