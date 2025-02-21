# %%
from sklearn.preprocessing import RobustScaler 

import numpy as np

import optuna

# from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import joblib

from Model import EIF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF

def tuning(model_name):
    """Tunes hyperparameters for a given anomaly detection model using Optuna.
    optuna.study.Study: The Optuna study object containing the results of the hyperparameter optimization.
    - The function defines an objective function for hyperparameter optimization using Optuna.
    - The objective function suggests hyperparameters based on the model_name and uses them to create the model.
    - The `outliers_fraction`, `X`, `y`, and `variables` are assumed to be predefined variables in the scope where this function is used.
    - The function defines a search space for each model and uses Optuna's GridSampler to perform grid search.
    - The optimization will end when the search spaces are exhausted or after 500 trials, whichever comes first.
    """
    def objective(trial, model_name):
        """
        Objective function for hyperparameter optimization using Optuna.
        Parameters:
        trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.
        model_name (str): The name of the model to optimize. Possible values are "LOF", "ABOD", "EIF", and "PCA".
        Returns:
        float: The evaluation score of the trained model.
        Notes:
        - The function suggests hyperparameters based on the model_name and uses them to create the model.
        - The function then trains the model using the `train` function and returns the evaluation score.
        - The `outliers_fraction`, `X`, and `variables` are assumed to be predefined variables in the scope where this function is used.
        """

        uB = 150
        lB = 10
        if model_name == "LOF":
            clf = LOF(
                n_neighbors=trial.suggest_int("n_neighbors", lB, uB),
                metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]),
                contamination=outliers_fraction
            )
        elif model_name == "ABOD":
            clf = ABOD(
                n_neighbors=trial.suggest_int("n_neighbors", lB, uB),
                contamination=outliers_fraction
            )
        elif model_name == "EIF":
            clf = EIF(
                contamination=outliers_fraction,
                ntrees=trial.suggest_int("ntrees", 100, 2000),
                extension_level=trial.suggest_int("extension_level", 0, X.shape[1]-1),
                seed=42,
                predictors=variables
            )
        elif model_name == "PCA":
            clf = PCA(
                contamination=outliers_fraction,
                n_selected_components=trial.suggest_int("n_selected_components", 2, X.shape[1])
            )
        


        return train(clf)

    def train(model):
        """
        Trains the given model using a transductive approach and calculates the average precision score.
        Args:
            model: The anomaly detection model to be trained. It should have a `fit` method and a `decision_scores_` attribute.
        Returns:
            float: The average precision score of the model on the training data.
        """

        # Performing in a transductive manner
        scaler = RobustScaler()

        # do five independent trials!
            # and report the average of the five trials
        # I am yet to see a paper that does not use cross-validation for this task

        X_train = scaler.fit_transform(X)
        model.fit(X_train)
        test_scores = model.decision_scores_
        y_test = y
        
        return average_precision_score(y_test, test_scores)

    # Grid search
    if model_name == "LOF":
        search_space = {'n_neighbors': list(range(10, 155, 5)),
                        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    elif model_name == "ABOD":
        search_space = {'n_neighbors': list(range(0, 155, 5))}
    elif model_name == "EIF":
        search_space = {'ntrees': list(range(100, 2000, 50)), 
                        'extension_level': list(range(0, X.shape[1]))}
    else:
        search_space = {'n_selected_components': list(range(2, X.shape[1]))}

    func = lambda trial: objective(trial, model_name)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")

    study.optimize(func, n_trials=500)

    return study

if __name__ == "__main__":
    data = joblib.load("results/training/data0_70.pkl")
    variables = data[0].loc[:, "confidence":].columns

    models = ["LOF", "ABOD", "EIF", "PCA"]

    for i in range(len(data)):
        y = np.array(data[i].loc[:, "Y"]).T 
        y = np.where(y == 'Outlier', 1, 0)

        X = np.array(data[i].loc[:, "confidence":])
        
        outliers_fraction = np.count_nonzero(y) / len(y) if np.count_nonzero(y) > 0 else 0.01
        for j in models:
            joblib.dump(tuning(j), f"results/hyperparameter/tuning_{j}_Orchard_{i+1}.pkl")