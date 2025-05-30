from sklearn.preprocessing import RobustScaler 

import numpy as np

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import joblib

from Model import EIF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF

import h2o

def tuning():
    """Tunes hyperparameters for a given anomaly detection model using Optuna.
    optuna.study.Study: The Optuna study object containing the results of the hyperparameter optimization.
    - The function defines an objective function for hyperparameter optimization using Optuna.
    - The objective function suggests hyperparameters based on the model_name and uses them to create the model.
    - The `outliers_fraction`, `X`, `y`, and `variables` are assumed to be predefined variables in the scope where this function is used.
    - The function defines a search space for each model and uses Optuna's GridSampler to perform grid search.
    - The optimization will end when the search spaces are exhausted or after 500 trials, whichever comes first.
    """
    def objective(trial):
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

        clf = EIF(contamination=outliers_fraction,
                  ntrees=trial.suggest_int("ntrees", 100, 2000),
                  extension_level=trial.suggest_int("extension_level", 0, X.shape[1]-1),
                  seed=42,
                  predictors=variables
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
        scores = []
        for i in range(5):
            if np.count_nonzero(y) > 1:
                X_train, X_test, _, y_test = train_test_split(X,
                                                              y,
                                                              test_size=0.2, 
                                                              stratify=y,
                                                              random_state=i)
            else:
                X_train, X_test, _, y_test = train_test_split(X,
                                                              y,
                                                              test_size=0.2, 
                                                              random_state=i)
        
            # standardizing data for processing
            scaler = RobustScaler()
            X_train_norm = scaler.fit_transform(X_train)
            scaler = RobustScaler()
            X_test_norm = scaler.fit_transform(X_test)
            
            # Fit the model
            model.fit(X_train_norm)
            
            # Predict the anomaly scores
            test_scores = model.decision_function(X_test_norm)

            scores.append(average_precision_score(y_test, test_scores))
        
        return np.mean(scores)

    # Grid search
    search_space = {'ntrees': list(range(100, 2000, 100)), 
                    'extension_level': list(range(0, X.shape[1]))}
    
    func = lambda trial: objective(trial)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")

    study.optimize(func, n_trials=500)

    return study

if __name__ == "__main__":
    data = joblib.load("/scratch/blnedw003/data0_70.pkl")
    variables = data[0].loc[:, "confidence":].columns

    for i in range(len(data)):
        y = np.array(data[i].loc[:, "Y"]).T 
        y = np.where(y == 'Outlier', 1, 0)

        X = np.array(data[i].loc[:, "confidence":])
        
        outliers_fraction = np.count_nonzero(y) / len(y) #if np.count_nonzero(y) > 0 else 0.01
        
        if outliers_fraction == 0:
            # no need to tune
            # when there is no outliers 
            # present
            continue


        joblib.dump(tuning(), f"results/hyperparameter/tuning_EIF_Orchard_{i+1}.pkl")
        h2o.shutdown()