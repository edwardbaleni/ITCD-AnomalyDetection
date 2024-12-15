# %%
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA
import utils.Triangulation as tri
import esda
from Model import EIF


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
from sklearn.model_selection import train_test_split

y = np.array(data.loc[:, "Y"]).T 
    # Change outlier to 1 and inlier to 0 in data
y = np.where(y == 'Outlier', 1, 0)

outliers_fraction = np.count_nonzero(y) / len(y)

X = np.array(data.loc[:, "confidence":]) 
# 60% data for training and 40% for testing
X_train, X_test, _, y_test = train_test_split(X,
                                            y,
                                            test_size=0.4,
                                            stratify=y,
                                            random_state=42)

# standardizing data for processing
X_train_norm = utils.engineer._scaleData(X_train)
X_test_norm = utils.engineer._scaleData(X_test)


# %%
clf = EIF(outliers_fraction, data.loc[:, "confidence":].columns)

# %%
clf.fit(X_train_norm)

# %%
y_pred = clf.decision_function(X_test_norm)

# %%

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_true=y_test, 
                                 y_pred=y_pred)#np.array(clf.decision_scores_))
plt.show()


# %%
# Compute ROC curve and ROC area
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)
print(f'Average precision-recall score: {average_precision:.2f}')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('foo.png')

# %%




# %%



from Model import Geary

X = utils.engineer._scaleData(X)
clf = Geary(contamination=outliers_fraction, 
            geometry=data["geometry"], 
            centroid=data["centroid"])
clf.fit(X)

test_scores = clf.decision_scores_
labels = clf.labels_


# %%
clf = EIF(outliers_fraction, data.loc[:, "confidence":].columns)
clf.fit(X)
test_scores = clf.decision_scores_
labels = clf.labels_

# %%

normal = data[labels == 0]
abnormal = data[labels == 1]

plotA.plot(tryout, normal, abnormal)
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
abnormal.plot(ax=ax, facecolor = 'none',edgecolor='blue')
plt.savefig('foo.png')


# %%

# just want to see if ABOD is indeed too slow
# default is way too slow!
from pyod.models.pca import PCA

clf = PCA(contamination=outliers_fraction)
clf.fit(X)
test_scores = clf.decision_scores_
labels = clf.labels_

normal = data[labels == 0]
abnormal = data[labels == 1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
abnormal.plot(ax=ax, facecolor = 'none',edgecolor='blue')
plt.savefig('foo2.png')




# %%
# TODO: https://github.com/jhmadsen/DDoutlier/blob/master/R/OutlierFunctionLibrary.R
#       Might not need to include thi just becuase it is not a popular method at all
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

def RDOS(dataset, k=5, h=1):
    dataset = np.array(dataset)
    n, d = dataset.shape

    if not isinstance(k, int):
        raise ValueError('k input must be numeric')
    if k >= n or k < 1:
        raise ValueError('k input must be less than number of observations and greater than 0')
    if not isinstance(h, (int, float)):
        raise ValueError('h input must be numeric')
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError('dataset input is not numeric')

    dist_matrix = squareform(pdist(dataset))
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)

    def func_dist(x1, x2):
        return len(np.intersect1d(x1, x2))

    sNN_matrix = np.array([[func_dist(indices[i], indices[j]) for j in range(n)] for i in range(n)])

    neighborhood = []
    for i in range(n):
        kNN = indices[i]
        rNN = np.where(indices == i)[0]
        sNN = np.where(sNN_matrix[i] > 0)[0]
        neighborhood.append(np.unique(np.concatenate((kNN, rNN, sNN))))

    px = np.zeros(n)
    for i in range(n):
        Kgaussian = (1 / ((2 * np.pi) ** (d / 2))) * np.exp(-(dist_matrix[i, neighborhood[i]] / (2 * h ** 2)))
        px[i] = (1 / (len(neighborhood[i]) + 1)) * np.sum((1 / (h ** d)) * Kgaussian)

    RDOS = np.zeros(n)
    for i in range(n):
        RDOS[i] = np.sum(px[neighborhood[i]]) / (len(neighborhood[i]) * px[i])

    return RDOS

a = RDOS(X, k=5, h=1)

_, ax = plt.subplots(1, figsize=(20, 20))
tryout.plot.imshow(ax=ax)
data.assign(cl= a).plot(column='cl', categorical=False,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='white', legend=True, alpha=0.7)
ax.set_axis_off()
plt.title("Anomaly Scores")
plt.savefig('foo3.png')















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