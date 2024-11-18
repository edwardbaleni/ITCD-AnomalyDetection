import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA
import utils.Triangulation as tri
import esda


class Geary:
    def __init__(self, contamination=0.5, geom=None, centroid=None):
        self.labels_ = None
        self.decision_scores_ = None
        self.contamination = contamination  
        self.geom = geom
        self.centroid = centroid

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
        """
        # local Geary C Statistic

        X["geometry"], X["centroid"] = self.geom, self.centroid

        d_w, _, _, _ = tri.delauneyTriangulation(X)

        # but we already know that delauney is better!
        w = d_w
        xx = X.loc[:, :"geometry"].values.T.tolist()
        xx = [pd.Series(x) for x in xx]
        lG_mv = esda.Geary_Local_MV(connectivity=w).fit(xx)

        self.decision_scores_ = np.log(lG_mv.localG)
        self.labels_ = np.where(np.log(lG_mv.localG) >= self.contamination * np.log(lG_mv.localG.max()), 1, 0)