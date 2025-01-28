# %%
import utils
import matplotlib.pyplot as plt
import numpy as np

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyod.models.lof import LOF
from sklearn.metrics import average_precision_score
import pandas as pd
import joblib


sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = utils.salientEngineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
							  True)

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

X = np.array(data.loc[:, "confidence":])
outliers_fraction = np.count_nonzero(y) / len(y)

# %%

def objective(trial):
    # Split the data
    # 60% data for training and 40% for testing
    X_train, X_test, _, y_test = train_test_split(X,
                                                  y,
                                                  test_size=0.4, 
                                                  stratify=y,
                                                  random_state=42)
    
    # standardizing data for processing
    X_train_norm = utils.engineer._scaleData(X_train)
    X_test_norm = utils.engineer._scaleData(X_test)
    # Define the model
    model = LOF(
        n_neighbors=trial.suggest_int("n_neighbors", 10, 500),
        contamination=outliers_fraction
    )
    
    # Fit the model
    model.fit(X_train_norm)
    
    # Predict the anomaly scores
    test_scores = model.decision_function(X_test_norm)
    
    # Calculate the accuracy
    ap = average_precision_score(y_test, test_scores)
    
    return ap

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=500)

# %% Can actually save study as a pkl file!
# Save the study

joblib.dump(study, "results/hyperparameter/optuna_study.pkl")

# %%

