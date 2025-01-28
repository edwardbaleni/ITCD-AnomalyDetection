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
img = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
img = img/255

y = np.array(data.loc[:, "Y"]).T 
    # Change outlier to 1 and inlier to 0 in data
y = np.where(y == 'Outlier', 1, 0)

outliers_fraction = 0.05#np.count_nonzero(y) / len(y)



# %% With the full feature set
data_full = myData.data.copy(deep=True)
data_full.loc[:,'confidence':] = utils.engineer._scaleData(data_full.loc[:, "confidence":])
from pyod.models.lof import LOF

clf = LOF(contamination=outliers_fraction)
clf.fit(data_full.loc[:, "confidence":])
test_scores = clf.decision_scores_
labels = clf.labels_

normal = data[labels == 0]
abnormal = data[labels == 1]
plotA.plot(img, normal, abnormal, 'data_full.png')


# %% Data Shape

data_sensitive = myData.data.copy(deep=True)
fin_data = list(data_sensitive.loc[:,:'roundness'].columns) + ["compactness", "convexity", "solidity", "bendingE",  "DSM" , "NDRE", "OSAVI", "ASM", "Corr"]+ list(data_sensitive.loc[:,"z1":'z24'])
data_sensitive = data_sensitive.loc[:, fin_data]
data_sensitive.loc[:,'confidence':] = utils.engineer._scaleData(data_sensitive.loc[:, "confidence":])


from pyod.models.lof import LOF

clf = LOF(contamination=outliers_fraction)
clf.fit(data_sensitive.loc[:, "confidence":])
test_scores = clf.decision_scores_
labels = clf.labels_

normal = data[labels == 0]
abnormal = data[labels == 1]

plotA.plot(img, normal, abnormal, 'data_sensitive.png')

# %%


# Obtain the precision for full dataset and precision for reduced dataset
# As well as images of improvements!

















# %%
from sklearn.model_selection import train_test_split


X = np.array(data.loc[:, "confidence":]) 

# 60% data for training and 40% for testing
X_train, X_test, _, y_test = train_test_split(X,
                                            y,
                                            test_size=0.4,
                                            stratify=y,
                                            random_state=42)

# %%
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