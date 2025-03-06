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


data = joblib.load("results/testing/data70_101.pkl")

data_local = data[-1]
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
    plt.figure(figsize=(12, 6))
    # Plot individual observation
    plt.plot(outliers.columns, outliers.loc[idx, :], marker='o', linestyle='-', alpha=0.7, label=f'Observation {idx}')
    
    # Plot 99th percentile line
    plt.plot(outliers.columns, percentile_99, marker='s', linestyle='--', color='red', linewidth=2, label='99th percentile')
    
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Observation {idx} vs 99th Percentile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/oos/interpretability/interpretability_Orchard_100_outlier_{idx}.png")
    plt.show()


# %%



    




# %%
