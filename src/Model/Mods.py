import numpy as np
import pandas as pd
import utils.Triangulation as tri
import esda


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
        """
        # local Geary C Statistic

        # X["geometry"] = self.geometry
        # X["centroid"] = self.centroid

        # w, _, _, _ = tri.delauneyTriangulation(X)

        # X.drop(columns=["geometry", "centroid"], inplace=True)
        w = self.w
        xx = X.T
        xx = [pd.Series(x) for x in xx]
        lG_mv = esda.Geary_Local_MV(connectivity=w).fit(xx)

        self.decision_scores_ = np.log(lG_mv.localG)
        self.labels_ = np.where(np.log(lG_mv.localG) >= self.contamination * np.log(lG_mv.localG.max()), 1, 0)