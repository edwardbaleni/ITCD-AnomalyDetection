import numpy as np
import pandas as pd
import utils.Triangulation as tri
import esda
from scipy.special import expit


class Geary:
    def __init__(self, contamination=0.5, geometry=None, centroid=None):
        """
        Initialize the proposed model with given parameters.
        Parameters
        ----------
        contamination : float, optional (default=0.5)
            The amount of contamination of the data set, i.e., the proportion 
            of outliers in the data set.
        geometry : DataFrame or None, optional (default=None)
            The geometry data to be used for the model.
        centroid : DataFrame or None, optional (default=None)
            The centroid data to be used for the model.
        Attributes
        ----------
        labels_ : array, shape (n_samples,)
            Labels of the data points after fitting the model.
        decision_scores_ : array, shape (n_samples,)
            The outlier scores of the data points after fitting the model.
        w : array
            The weights obtained from the Delaunay triangulation.
        """

        self.labels_ = None
        self.decision_scores_ = None
        self.contamination = contamination  
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

    
    # TODO: We can make Geary interpretable
    #       Following COPOD paper. We can demonstrate the main contributors to the outlier score
    #       by looking at the weighted square difference of the anomolous observation
    #       We can thereafter plot the value given for each attribute. This should illustrate quite clearly
    #       what is causing the observation to be anomolous. High point values will be the main contributors.
    #       This is possible due to the additive nature of the Geary C statistic.
    #       If I copy the code from https://pysal.org/esda/_modules/esda/geary.html#Geary and not sum the values
    #       then I should be able to get the individual contributions to the Geary C statistic. 