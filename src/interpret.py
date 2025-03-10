# XXX: This file is used to capture the interpretability of the model.
# Specifically, it should demonstrate which features are the greatest drivers of anomalies!

# The univariate Geary C statistic over each feature is a good start!
# Using this we can see which features explain the outlierness more!

# %%
# XXX: Interpretability!
from esda import Geary_Local
import utils.Triangulation as tri
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd

import joblib
import pandas as pd
from Model import Geary

from sklearn.preprocessing import RobustScaler
import numpy as np


# data = joblib.load("results/testing/data70_101.pkl")
data = joblib.load("results/training/data0_70.pkl")

# Orchard 62 is data 61


# %%
data_local = data[61]
geometry = data_local["geometry"]
centroid = data_local["centroid"]

scaler = RobustScaler()
data_local.loc[:,'confidence':] = scaler.fit_transform(data_local.loc[:, 'confidence':])

ww , _, _, _ = tri.delauneyTriangulation(pd.concat([geometry, centroid], axis=1))

cols = list(data_local.columns)[5:]
uni_Geary = pd.DataFrame()
for i in range(len(cols)):

    X = data_local.loc[:,cols[i]]
    lG_mv = Geary_Local(connectivity=ww).fit(X)
    
    # Transform scores
    centerScore = lG_mv.localG - np.mean(lG_mv.localG)
    probs = expit(centerScore)
    uni_Geary[cols[i]] = probs

clf = Geary(contamination=0.05,
                geometry=data_local["geometry"], 
                centroid=data_local["centroid"])
X = np.array(data_local.loc[:, 'confidence':])
clf.fit(X)
y_test_scores = clf.decision_scores_
y_labels = clf.labels_


outliers = uni_Geary[y_labels == 1]
# outliers.reset_index(drop=True, inplace=True)
import matplotlib.pyplot as plt


# Plot each observation (y-axis) versus the dimensions (x-axis)
# First calculate the 99th percentile for each dimension
percentile_99 = outliers.quantile(0.99, axis=0)

# Plot individual observations and the 99th percentile line
for idx in outliers.index:
    plt.figure(figsize=(20, 12))
    # Plot individual observation
    plt.plot(outliers.columns, outliers.loc[idx, :], marker='o', linestyle='-', alpha=0.7, linewidth=4, markersize=10, label=f'Attribute Outlier Score')
    
    # Plot 99th percentile line
    plt.plot(outliers.columns, percentile_99, marker='s', linestyle='--', color='red', linewidth=4, markersize=10, label='99th percentile')
    
    plt.xlabel('Attribute', fontsize=30)
    plt.ylabel('Univariate Geary\'s C Statistic', fontsize=30)
    # plt.title(f'Observation {idx} vs 99th Percentile')
    plt.legend(fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=25, rotation=45)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.savefig(f"results/oos/interpretability/interpretability_Orchard_61_outlier_{idx}.png")
    plt.show()


# %%

# XXX: Need to plot the individual outlier
from utils.plotAnomaly import plot
import geopandas as gpd

images = joblib.load("results/training/images60_70.pkl")

images = images[1]

# Check which y_labels are equal to 1
outlier_indices = np.where(y_labels == 1)[0]

for i in outlier_indices:
    plot(images, data_local, gpd.GeoDataFrame(pd.DataFrame(data_local.iloc[i]).T, geometry='geometry'), f"results/oos/interpretability/geary_{i}.png")
    # img = images
    # normal = data_local[y_labels == 0]
    # anomaly = 
    # fig, ax = plt.subplots(figsize=(20, 20))
    # img.plot.imshow(ax=ax)
    # ax.axis('off')
    # normal.plot(ax=ax, edgecolor='red', label='Normal Regions') 
    # anomaly.plot(ax=ax,label='Anomaly Regions')
    # # custom_lines = [Line2D([0], [0], color='red', lw=2),
    # #                 Line2D([0], [0], color='blue', lw=2)]
    # # ax.legend(custom_lines, ['Normal', 'Outliers'], loc='upper right', fontsize=25)
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.title("")  # Data Geometries Colored by Y
    # fig.savefig(f"results/oos/interpretability/geary_{i}.png")
    # plt.show()

plot(images, data_local[y_labels==0], data_local[y_labels==1], f"results/oos/interpretability/geary_full.png")


# %%
    




# %%
