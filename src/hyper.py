# %%
import utils
import matplotlib.pyplot as plt
import numpy as np

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import pandas as pd
import joblib

from Model import EIF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF

import warnings
warnings.filterwarnings("ignore")



sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = utils.salientEngineer(num, 
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

X = np.array(data.loc[:, "confidence":])
outliers_fraction = np.count_nonzero(y) / len(y)

# %%

def tuning(model_name):
    def objective(trial, model_name):
        if model_name == "LOF":
            clf = LOF(
                n_neighbors=trial.suggest_int("n_neighbors", 10, 500),
                contamination=outliers_fraction
            )
        elif model_name == "ABOD":
            clf = ABOD(
                n_neighbors=trial.suggest_int("n_neighbors", 10, 70),
                contamination=outliers_fraction
            )
        elif model_name == "EIF":
            clf = EIF(
                contamination=outliers_fraction,
                ntrees=trial.suggest_int("ntrees", 100, 1000),
                extension_level=trial.suggest_int("extension_level", 0, X.shape[1]-1), # 7 is the maximum extension level
                seed=42,
                predictors=data.loc[:, "confidence":].columns
            )
        elif model_name == "PCA":
            clf = PCA(
                contamination=outliers_fraction,
                n_selected_components=trial.suggest_int("n_selected_components", 2, X.shape[1]),
            )
        


        return train(clf)

    def train(model):
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
        
        # Fit the model
        model.fit(X_train_norm)
        
        # Predict the anomaly scores
        test_scores = model.decision_function(X_test_norm)
        
        # Calculate the accuracy
        ap = average_precision_score(y_test, test_scores)
        
        return ap


    func = lambda trial: objective(trial, model_name)
    study = optuna.create_study(direction="maximize")

    # TODO: If this runs on HPC, increase the number of trials
    study.optimize(func, n_trials=100)

    return study

# %%
# Save the study
models = ["ABOD", "LOF", "EIF", "PCA"]
# TODO: make it such that the loop works over all 30 datasets!
for i in models:
    for j in range(5):
        joblib.dump(tuning(i), f"results/hyperparameter/tuning_{i}_Orchard_{j}.pkl")

# %%

