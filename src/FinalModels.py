# %%
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA
import utils.Triangulation as tri
import esda
import Model

sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = utils.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
							  False)
data = myData.data.copy(deep=True)
delineations = myData.delineations.copy(deep=True)
mask = myData.mask.copy(deep=True)
spectralData = myData.spectralData
erf_num = myData.erf
refData = myData.ref_data.copy(deep=True)
# For plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255

# %%
# local Geary C Statistic

d_w, d_g, d_p, v_cells = tri.delauneyTriangulation(data)
knn_w, knn_g, knn_p, knn_centroids = tri.KNNGraph(data)

    # Plot Triangulations
tri.delauneyPlot(d_g, d_p, v_cells, tryout, True)
tri.KNNPlot(knn_g, knn_p, knn_centroids, tryout, True)

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

from scipy.special import expit

centerScore = lG_mv.localG - np.mean(lG_mv.localG)
probs = expit(centerScore)



anomaly = data[probs >= 0.98]
nominal = data[probs < 0.98]

plotA.plot(tryout, nominal, anomaly)

plotA.plotScores(tryout, data, probs)





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