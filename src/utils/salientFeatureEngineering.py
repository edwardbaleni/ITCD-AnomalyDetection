# %%

import shapely.plotting
import shapely
import numpy as np
import pandas as pd
import math

# parent class
from utils.dataCollect import collect
from sklearn.preprocessing import RobustScaler

from rasterstats import zonal_stats

import cv2 as cv
import mahotas as mh
import skimage

import warnings
warnings.filterwarnings("ignore")


class salientEngineer(collect):

    def __init__(self, num, tifs, geojsons, zips, scale = True):
        super().__init__(num, tifs, geojsons, zips)
        self.scaleData(self.delineations, self.spectralData, scale)
        self.labelRefData(self.data, self.ref_data, self.mask)
    
    def labelRefData(self, data, refData, mask):
        """
        Labels the reference data based on the provided data.
        Parameters:
        data (GeoDataFrame): The estimated delineations with geometries.
        refData (GeoDataFrame): The reference data with geometries.
        mask (GeoDataFrame): The mask to apply on the reference data.
        Returns:
        None: The function updates the 'data' attribute with labeled data and 'ref_data' attribute with masked reference data.
        The function performs the following steps:
        1. Masks the reference data using the provided mask.
        2. Converts the CRS of the reference data to match the data CRS.
        3. Computes the centroid of the reference data polygons.
        4. Checks for under-segmentation and false positives by counting the number of reference centers within estimated delineations.
        5. Checks for over-segmentation by counting the number of estimated centers within the reference delineations.
        6. Labels the data as "TP" (True Positive) by default.
        7. Updates the labels to "Outlier" for over-segmented and under-segmented/false positive cases.
        8. Relocates the 'Y' column to a position after the 'centroid' column in the data dataframe.
        """
        # Need to mask refData
        refData.to_crs(data.crs, inplace=True)
        index_mask_intersect = collect._recursivePointRemoval(refData, mask)
        refData = refData.iloc[index_mask_intersect]
        refData.reset_index(drop=True, inplace=True)
        refData.to_crs("3857", inplace=True)
        refData.loc[:,"centroid"] = refData["geometry"].centroid
        refData.to_crs(data.crs, inplace=True)
        refData.loc[:, "centroid"] = refData.loc[:,"centroid"].to_crs(data.crs)
        underSeg_Fp = data['geometry'].apply(lambda x: refData["centroid"].within(x).sum())
  
        overSeg = []
        for _, row in data.iterrows():
            count = refData["geometry"].apply(lambda x: row["centroid"].within(x)).sum()
            overSeg.append(count)

        data['Y'] = "TP"

        # Over-segmentation
        for i, key in enumerate(overSeg):
            if key > 1:
                # Over
                data.loc[i, 'Y'] = "Outlier"
        # Under-segmentation and False Positives
        for i, key in enumerate(underSeg_Fp):
            if key == 0:
                # Fp
                data.loc[i, 'Y'] = "Outlier"
            if key > 1:
                # Under
                data.loc[i, 'Y'] = "Outlier"

        cols = list(data.columns)
        cols.insert(cols.index('centroid') + 1, cols.pop(cols.index('Y')))
        data = data[cols]

        self.data = data
        self.ref_data = refData


    @staticmethod
    def _zernickeMoments(im):
        """
        Calculate the Zernike moments for a given image.
        Parameters:
        im (ndarray): The input image for which Zernike moments are to be calculated.
        Returns:
        list: A list of Zernike moments for the input image.
        Notes:
        This function uses the `zernike_moments` function from the `mahotas` library to compute the moments.
        The radius is calculated as half of the maximum bounding box dimension of the labeled image.
        The center of mass of the image is used as the center for the Zernike moments calculation.
        """
        zernike_moments = mh.features.zernike_moments(im, 
                                                      radius = (mh.labeled.bbox(im).max()/2)
                                                      )
        return list(zernike_moments)[1:]


    @staticmethod
    def _texture(im):
        """
        Calculate texture features from a grayscale image using the gray-level co-occurrence matrix (GLCM).
        Parameters:
        im (ndarray): Input grayscale image.
        Returns:
        list: A list containing the following texture features:
            - contrast: Measure of the intensity contrast between a pixel and its neighbor over the whole image.
            - correlation: Measure of how correlated a pixel is to its neighbor over the whole image.
            - ASM (Angular Second Moment): Measure of the uniformity or energy of the image.
        Notes:
        - This function uses the `graycomatrix` and `graycoprops` functions from the `skimage.feature` module.
        - The distances parameter in `graycomatrix` is set to [2], which considers pixels that are 2 units apart.
        - The angles parameter in `graycomatrix` is set to [0], which considers horizontal pixel pairs.
        - The levels parameter in `graycomatrix` is set to 256, which is suitable for 8-bit images.
        - The symmetric and normed parameters in `graycomatrix` are set to True.
        - The function calculates contrast, correlation, and ASM from the co-occurrence matrix.
               
        Notes : # this get the first geometry
                # we can use this to get texture properties
                # Pyfeat library is good. But this one is more trusted
                # Need to play around with 
                #         # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
        """
        co_matrix = skimage.feature.graycomatrix(im, 
                                            distances=[2],
                                            angles=[0], 
                                            levels=256, 
                                            symmetric=True, 
                                            normed=True)
        
        correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0][0]
        ASM = skimage.feature.graycoprops(co_matrix, 'ASM')[0][0]

        return [correlation, ASM]

    def imageA(self, dat, spectral):
        """
        Processes an image by clipping it to a specified region and converting it to grayscale,
        then extracts texture and Zernike moments features.
        Parameters:
        dat (GeoDataFrame): The geographical data used to clip the image.
        spectral (xarray.DataArray): The spectral image data to be processed.
        Returns:
        pd.Series: A pandas Series containing the extracted Zernike moments and texture features.
        """
        touch = spectral.rio.clip([dat], spectral.rio.crs)
        im = cv.cvtColor(touch.T.to_numpy()[:,:,:3], cv.COLOR_BGR2GRAY).T

        text = self._texture(im)
        zernicke = self._zernickeMoments(im)
        return pd.Series(zernicke + text)
    
    @staticmethod
    def _radiusOfGyration(xx_centre, xx, yy):
        """
        Calculate the radius of gyration for a set of points.
        The radius of gyration is a measure of the distribution of the points around a central point.
        Parameters:
        xx_centre (shapely.geometry.Point): The central point from which distances are measured.
        xx (list or array-like): The x-coordinates of the points.
        yy (list or array-like): The y-coordinates of the points.
        Returns:
        pd.Series: The radius of gyration.
        """
        rad = 0
        for i in range(len(xx)):
            dist = xx_centre.distance(shapely.Point(xx[i], yy[i]))
            rad += dist**2

        return pd.Series(math.sqrt(rad) / len(xx))

    @staticmethod
    def _major_minor(xx):
        """
        Calculate the lengths of the major and minor axes of the minimum bounding rectangle of a given shape.
        Args:
            xx (shapely.geometry.Polygon): A shapely Polygon object.
        Returns:
            pd.Series: A pandas Series containing the lengths of the minor and major axes.
        """
        mbr_points = list(zip(*xx.minimum_rotated_rectangle.exterior.xy))
        mbr_lengths = [shapely.LineString([mbr_points[i], mbr_points[i + 1]]).length for i in range(len(mbr_points) - 1)]
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)
        return pd.Series([minor_axis, major_axis])

    @staticmethod
    def _curvature(xx):
        """
        Calculate the curvature of a 2D curve given its coordinates.
        Parameters:
        xx (numpy.ndarray): A 2D array of shape (n, 2) where n is the number of points, 
                            and each row represents the (x, y) coordinates of a point.
        Returns:
        numpy.ndarray: A 1D array of curvature values for each point on the curve.
        Notes:
        The curvature is calculated using the formula:
        curvature = |(d2x/dt2 * dy/dt - dx/dt * d2y/dt2)| / (dx/dt^2 + dy/dt^2)^(3/2)
        Reference:
        https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        """
        dx_dt = np.gradient(xx[:, 0])
        dy_dt = np.gradient(xx[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature

    @staticmethod
    def _bendingEnergy(xx, r):
        """
        Calculate the bending energy of a polygon.

        Parameters:
        xx (Polygon): The polygon for which the bending energy is calculated.
        r (float): A radius value used in the bending energy calculation.

        Returns:
        float: The bending energy of the polygon.

        Notes:
        - The function uses the curvature of the polygon's coordinates.
        - The Shapely library is used to get the coordinates of the polygon.
        - The bending energy is calculated using the formula:
          max((2 * math.pi) / r, 1 / L * sum(curvature(xx)**2)),
          where L is the number of coordinates in the polygon.
        """
        import math
        xx = shapely.get_coordinates(xx)
        L = xx.shape[0]
        return max( (2 * math.pi)/r , 1/L * sum( salientEngineer._curvature(xx)**2 ))

    def shapeDescriptors(self, placeholder):
        """
        Calculate various shape descriptors for a given GeoDataFrame.
        Parameters:
        placeholder (GeoDataFrame): A GeoDataFrame containing geometries for which shape descriptors are to be calculated.
        Returns:
        GeoDataFrame: The input GeoDataFrame with additional columns for each calculated shape descriptor.
        Shape Descriptors:
        - crown_projection_area: Area of the geometry.
        - crown_perimeter: Perimeter of the geometry.
        - radius_of_gyration: Radius of gyration of the geometry.
        - minor_axis: Length of the minor axis of the geometry.
        - major_axis: Length of the major axis of the geometry.
        - roundness: Roundness of the geometry.
        - circularity: Circularity of the geometry.
        - shape_index: Shape index of the geometry.
        - form_factor: Form factor of the geometry.
        - compactness: Compactness of the geometry.
        - convexity: Convexity of the geometry.
        - solidity: Solidity of the geometry.
        - elongation: Elongation of the geometry.
        - bendingE: Bending energy of the geometry.
        Notes: https://iopscience.iop.org/article/10.1088/1361-6560/abfbf5/data
        """
        placeholder =  placeholder.to_crs(3857)
        convexHull = placeholder.loc[:, "geometry"].convex_hull
        convex_area = shapely.area(convexHull)
        convex_perimeter = shapely.length(convexHull)
        
        area = shapely.area(placeholder.loc[:,"geometry"])
        perimeter = shapely.length(placeholder.loc[:,"geometry"])
        placeholder["radius_of_gyration"] = placeholder[["centroid", "geometry"]].apply(lambda x: salientEngineer._radiusOfGyration(x.iloc[0], x.iloc[1].exterior.coords.xy[0], x.iloc[1].exterior.coords.xy[1]), axis=1)
        placeholder[["minor_axis", "major_axis"]] = placeholder["geometry"].apply(lambda x: salientEngineer._major_minor(x))
        placeholder["roundness"] = (4 * area) / (math.pi * (placeholder["major_axis"]**2))
        placeholder["circularity"] = 4 * math.pi * (area) / (perimeter**2)
        placeholder["compactness"] = (perimeter**2) / (4 * math.pi * area)
        placeholder["convexity"] = perimeter / convex_perimeter
        placeholder["solidity"] = area / placeholder["geometry"].convex_hull.area
        placeholder["bendingE"] = list(map(salientEngineer._bendingEnergy, placeholder["geometry"], placeholder["radius_of_gyration"]))
        
        placeholder.drop(["minor_axis", "major_axis", "radius_of_gyration"], axis=1, inplace = True)
        
        return placeholder

    @staticmethod
    def _detStats(x, geometry):
        """
        Calculate zonal statistics for a given geometry and raster data.
        Parameters:
        x (xarray.DataArray): The raster data array.
        geometry (geopandas.GeoDataFrame): The geometry for which to calculate the statistics.
        Returns:
        pandas.DataFrame: A DataFrame containing the calculated zonal statistics (mean).
        """

        geometry = geometry.to_crs(x.rio.crs)
        affine = x.rio.transform()
        array = np.array(x)[0]

        return pd.DataFrame(zonal_stats(geometry, array, nodata=np.nan,
                    affine=affine,
                    stats="mean"))
        

    # TODO: Select a better statistic mean/median/mode/max/etc.
    # TODO: Speed up zonal statistics
    def zonalStatistics(self, placeholder, spectral):
        """
        Computes zonal statistics for various spectral features and adds them to the placeholder DataFrame.
        Parameters:
        placeholder (pandas.DataFrame): DataFrame containing geometries for which the statistics are computed.
        spectral (dict): Dictionary containing spectral data arrays with keys such as 'dem', 'nir', 'ndre', 'ndvi', 'gndvi', 'savi', 'evi', and 'osavi'.
        Returns:
        pandas.DataFrame: Updated DataFrame with computed mean values for each spectral feature.
        """
        geom = placeholder.loc[:,"geometry"]

        placeholder[["DSM"]] = salientEngineer._detStats(spectral["dem"], geom)
        placeholder[["NDRE"]] = salientEngineer._detStats(spectral["ndre"], geom)
        placeholder[["OSAVI"]] = salientEngineer._detStats(spectral["osavi"], geom)
        
        return placeholder

    @staticmethod
    def _scaleData(x):
        """
        Scales the input data using the RobustScaler.
        Parameters:
            x (array-like): The data to be scaled.
        Returns:
            array-like: The scaled data.
        """
                
        scaler = RobustScaler()
        return(scaler.fit_transform(x))


    def scaleData(self, placeholder, spectral, scale):
        """
        Scales and processes spatial data with various feature engineering techniques.
        Parameters:
        -----------
        placeholder : GeoDataFrame
            A GeoDataFrame containing spatial data with a 'geometry' column.
        spectral : dict
            A dictionary containing spectral data, including 'rgb' key for RGB image data.
        scale : bool
            A boolean flag indicating whether to apply feature scaling.
        Returns:
        --------
        None
            The processed data is stored in the instance variable `self.data` and delineations in `self.delineations`.
        Notes:
        ------
        - The method calculates centroids, latitude, and longitude for the geometries.
        - Spatial features are processed first, followed by shape descriptors and zonal statistics.
        - The coordinate reference system (CRS) is converted to 4326 for zonal statistics.
        - Additional analysis features are computed using the `imageA` method.
        - If `scale` is True, robust scaling is applied to the features starting from 'confidence' column.
        """
        
        placeholder["centroid"] = shapely.centroid(placeholder.loc[:,"geometry"])
        placeholder = placeholder.loc[:, ['geometry', 'centroid', 'confidence']]
        placeholder = self.shapeDescriptors(placeholder)
        placeholder = placeholder.to_crs(4326) 
        placeholder = self.zonalStatistics(placeholder, spectral)
        analysis = ["z" + str(x) for x in range(1,25)] + ['Corr', 'ASM']
        placeholder[analysis] = placeholder["geometry"].apply(lambda x: self.imageA(x, spectral['rgb']))

        if (scale):
            placeholder.loc[:, "confidence":] = salientEngineer._scaleData(placeholder.loc[:,'confidence':])
        
        self.data = placeholder
        self.delineations = placeholder[["geometry"]]