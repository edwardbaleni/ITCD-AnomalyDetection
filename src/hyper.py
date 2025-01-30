# %%
import utils
import numpy as np

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import joblib

from Model import EIF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF

import warnings
warnings.filterwarnings("ignore")

data = joblib.load("results/training/data0_40.pkl")

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

if __name__ == "__main__":
    models = ["LOF", "ABOD", "EIF", "PCA"]

    for i in range(len(data)):
        y = np.array(data[i].loc[:, "Y"]).T 
        y = np.where(y == 'Outlier', 1, 0)

        X = np.array(data[i].loc[:, "confidence":])
        outliers_fraction = np.count_nonzero(y) / len(y)
        for j in models:
            joblib.dump(tuning(j), f"results/hyperparameter/tuning_{j}_Orchard_{i+1}.pkl")