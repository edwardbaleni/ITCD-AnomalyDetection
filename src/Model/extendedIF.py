import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import numpy as np
import pandas as pd

class EIF:
    def __init__(self, contamination=0.5, predictors=None):
        self.eif = None
        self.decision_scores_ = None
        self.labels_ = None
        self.mean_length = None
        self.h2o_df = None
        self.contamination = contamination
        self.predictors = list(predictors)

    def _setLabels(self, probs):
        self.labels_ = np.where(probs >= (1 - self.contamination), 1, 0)

    def _setDecisionScores(self, results):
        self.decision_scores_ = np.array(results["anomaly_score"])
        self.mean_length = np.array(results["mean_length"])

    @staticmethod
    def _convertFrame(data):
        return data.as_data_frame()

    def fit(self, X):
        h2o.init()
        X = pd.DataFrame(X)
        X.columns = self.predictors

        self.h2o_df = h2o.H2OFrame(X)

        eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                                    ntrees = 100,
                                                    sample_size = 256,#int(X.shape[0] * 0.8),
                                                    extension_level = 6)#len(predictors) - 1)
        
        eif.train(x = self.predictors,
                  training_frame = self.h2o_df)
        
        eif_result = EIF._convertFrame(eif.predict(self.h2o_df))
        
        self._setDecisionScores(eif_result)

        self._setLabels(self.decision_scores_)
        
        self.eif = eif


    def decision_function(self, X_test):
        X_test = pd.DataFrame(X_test)
        X_test.columns = self.predictors
        h20_test = h2o.H2OFrame(X_test)

        eif_result = EIF._convertFrame(self.eif.predict(h20_test))
        self._setDecisionScores(eif_result)
        self._setLabels(self.decision_scores_)

        return self.decision_scores_