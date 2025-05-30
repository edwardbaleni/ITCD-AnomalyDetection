import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import numpy as np
import pandas as pd

class EIF:
    def __init__(self, contamination=0.5, ntrees = 100, extension_level = 6, seed=42, predictors=None):
        self.ntrees = ntrees
        self.extension_level = extension_level
        self.seed = seed
        self.eif = None
        self.decision_scores_ = None
        self.labels_ = None
        self.mean_length = None
        self.h2o_df = None
        self.contamination = contamination
        self.predictors = list(predictors)

    # TODO: Check if labels are chosen based on scale from 0 to 1 
    #       or if it is based on the percntile of the anomaly scores
    #       Based on that will decide whether we include this method or not
    # def _setLabels(self, probs):
    #     self.labels_ = np.where(probs >= (1 - self.contamination), 1, 0)
    def _setLabels(self, probs, threshold):

        # Order the probabilities and assign values
        ordered_indices = np.argsort(probs)

        labels = np.zeros(len(probs))

        labels[-threshold:] = 1

        ordered_labels = np.zeros(len(probs))

        for i in range(len(probs)):
            ordered_labels[ordered_indices[i]] = labels[i]

        self.labels_ = ordered_labels


    def _setDecisionScores(self, results):
        self.decision_scores_ = np.array(results["anomaly_score"])
        self.mean_length = np.array(results["mean_length"])

    @staticmethod
    def _convertFrame(data):
        return data.as_data_frame()

    def fit(self, X):
        h2o.init()
        h2o.no_progress()
        X = pd.DataFrame(X)
        X.columns = self.predictors

        self.h2o_df = h2o.H2OFrame(X)

        eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                                    ntrees = self.ntrees,
                                                    sample_size = int(X.shape[0] * 0.6), # 256 is the defualt, but not using it just incase an orchard is small!
                                                    extension_level = self.extension_level, #len(predictors) - 1)
                                                    seed=self.seed)
        
        eif.train(x = self.predictors,
                  training_frame = self.h2o_df)
        
        eif_result = EIF._convertFrame(eif.predict(self.h2o_df))
        
        self._setDecisionScores(eif_result)

        samples = X.shape[0]
        self._threshold = int(round(samples * self.contamination, 0))

        self._setLabels(self.decision_scores_, self._threshold)
        # self._setLabels(self.decision_scores_)
        
        self.eif = eif


    def decision_function(self, X_test):
        X_test = pd.DataFrame(X_test)
        X_test.columns = self.predictors
        h20_test = h2o.H2OFrame(X_test)

        eif_result = EIF._convertFrame(self.eif.predict(h20_test))
        self._setDecisionScores(eif_result)
        # self._setLabels(self.decision_scores_)
        samples = X_test.shape[0]
        self._threshold = int(round(samples * self.contamination, 0))

        self._setLabels(self.decision_scores_, self._threshold)

        return self.decision_scores_