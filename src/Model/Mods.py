import numpy as np
import pandas as pd
import utils.Triangulation as tri
import esda
from scipy.special import expit


class Geary:
    def __init__(self, contamination=0.5, geometry=None, centroid=None):
        self.labels_ = None
        self.decision_scores_ = None
        self.contamination = 1 - contamination  
        self.w , _, _, _ = tri.delauneyTriangulation(pd.concat([geometry, centroid], axis=1))

    def fit(self, X):
        """
        Performs Geary's C Local Multivariate Spatial Autocorrelation on the given data.
        Parameters:
        data : DataFrame
            The DataFrame containing the data to be analyzed.
        tryout : GeoDataFrame
            The GeoDataFrame containing the data to be plotted.
        Returns:
        None
        Notes:
            Need to center scores about zero
            then use invrese logit transformation to get scores between 0 and 1
            otherwise, scores will be from 0.5 to 1
        """
        w = self.w
        xx = X.T
        xx = [pd.Series(x) for x in xx]
        lG_mv = esda.Geary_Local_MV(connectivity=w).fit(xx)

        centerScore = lG_mv.localG - np.mean(lG_mv.localG)
        probs = expit(centerScore)

        self.decision_scores_ = probs
        self.labels_ = np.where(probs >= (1 - self.contamination), 1, 0)