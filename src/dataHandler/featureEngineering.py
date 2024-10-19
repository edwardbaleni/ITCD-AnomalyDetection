# %%

import shapely.plotting
import shapely
# import geopandas as gpd
# import rasterio as rio
# from rasterio.plot import show
# from osgeo import ogr, gdal
# from osgeo import gdalconst
# from rasterio.mask import mask
# import earthpy.spatial as es
# import earthpy.plot as ep
# import earthpy as et
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


from sklearn.neighbors import NearestNeighbors as KNN
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
# parent class
from dataHandler.dataCollect import collect
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from rasterstats import zonal_stats

import cv2 as cv
import mahotas as mh
import skimage


class engineer(collect):

    def __init__(self, num, tifs, geojsons, zips, scale = True):
        super().__init__(num, tifs, geojsons, zips)
        self.scaleData(self.delineations, self.spectralData, scale)
        

                        # Feature Engineering

        # TODO: Need to remove radius, major and minor axis, crown projection area, crown perimeter,
        #       radius of gyration, short, long as they are not robust features
        #       Small trees are still trees and should be identified as such

        # can remove confidence but don't necessarily have to as it is a feature
        # that we can expext most datasets to have. Aerobotics provides this feature
        # with every dataset.

    # Radius of gyration - https://www.tutorialspoint.com/radius-of-gyration
    # For now we will use the straight line distance from the centroid to the vertices
    # However, in future this line should be robust, say in the case of a concave polygon
    # the line should not exit the polygon.
    # I want to calculate the distance between two points within a concave shape, 
    # and the distance line between these points cannot exit the polygon.

    @staticmethod
    def _radiusOfGyration(xx_centre, xx, yy):
        rad = 0
        for i in range(len(xx)):
            dist = xx_centre.distance(shapely.Point(xx[i], yy[i]))
            rad += dist**2

        return pd.Series(math.sqrt(rad) / len(xx))

    @staticmethod
    # https://stackoverflow.com/questions/13536209/efficient-way-to-measure-region-properties-using-shapely
    def _major_minor(xx):
        # major and minor axis
        mbr_points = list(zip(*xx.minimum_rotated_rectangle.exterior.xy))
        mbr_lengths = [shapely.LineString([mbr_points[i], mbr_points[i + 1]]).length for i in range(len(mbr_points) - 1)]
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)
        return pd.Series([minor_axis, major_axis])

    @staticmethod
    def _curvature(xx):
        """
        xx - coordinates

        Notes: https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        """
        dx_dt = np.gradient(xx[:, 0])
        dy_dt = np.gradient(xx[:, 1])
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature

    @staticmethod
    def _bendingEnergy(xx, r):
        """
        xx - Polygon
        """
        import math
        xx = shapely.get_coordinates(xx)
        L = xx.shape[0]
        return max( (2 * math.pi)/r , 1/L * sum( engineer._curvature(xx)**2 ))

    # TODO: Zernicke Moments

    @staticmethod
    def _zernickeMoments(im):
        zernike_moments = mh.features.zernike_moments(im, radius = (mh.labeled.bbox(im).max()/2),#(data["major_axis"][0]/2),
                                                        cm=mh.center_of_mass(im))
        return list(zernike_moments)
        # plt.imshow(im, interpolation='nearest')
        # plt.show()


    @staticmethod
    def _texture(im):
        """
        
        Notes : # this get the first geometry
                # we can use this to get texture properties
                # Pyfeat library is good. But this one is more trusted
                # Need to play around with 
        """
        co_matrix = skimage.feature.graycomatrix(im, 
                                            distances=[5], # looks at distances and how many pixels away to consider
                                            angles=[0], 
                                            levels=256, 
                                            symmetric=True, 
                                            normed=True)
        
        # Calculate texture features from the co-occurrence matrix
        # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
        contrast = skimage.feature.graycoprops(co_matrix, 'contrast')[0][0]
        correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0][0]
        energy = skimage.feature.graycoprops(co_matrix, 'energy')[0][0]
        homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')[0][0]
        ASM = skimage.feature.graycoprops(co_matrix, 'ASM')[0][0]
        diss = skimage.feature.graycoprops(co_matrix, 'dissimilarity')[0][0]

        return [contrast, correlation, energy, homogeneity, ASM, diss]


    def imageA(self, dat, spectral):
        """
        
        
        """
        touch = spectral.rio.clip([dat], spectral.rio.crs)
        im = cv.cvtColor(touch.T.to_numpy()[:,:,:3], cv.COLOR_BGR2GRAY).T

        text = self._texture(im)
        zernicke = self._zernickeMoments(im)
        return pd.Series(zernicke + text)



    # TODO: https://iopscience.iop.org/article/10.1088/1361-6560/abfbf5/data
    def shapeDescriptors(self, placeholder):
        placeholder =  placeholder.to_crs(3857)
        convexHull = placeholder.loc[:, "geometry"].convex_hull
        convex_area = shapely.area(convexHull)
        convex_perimeter = shapely.length(convexHull)
        # Shape Descriptors Not Robust
        placeholder["crown_projection_area"] = shapely.area(placeholder.loc[:,"geometry"])
        placeholder["crown_perimeter"] = shapely.length(placeholder.loc[:,"geometry"])
        placeholder["radius_of_gyration"] = placeholder[["centroid", "geometry"]].apply(lambda x: engineer._radiusOfGyration(x.iloc[0], x.iloc[1].exterior.coords.xy[0], x.iloc[1].exterior.coords.xy[1]), axis=1)
        # Robust Shape Descriptors
        placeholder[["minor_axis", "major_axis"]] = placeholder["geometry"].apply(lambda x: engineer._major_minor(x))
        placeholder["roundness"] = (4 * math.pi * placeholder["crown_projection_area"]) / (convex_perimeter**2)#(4 * placeholder["crown_projection_area"]) / (math.pi * placeholder["major_axis"]**2)
        # Circularity is NB for some reason
        placeholder["circularity"] = (placeholder["crown_perimeter"]**2) / (4 * math.pi * placeholder["crown_projection_area"])
        placeholder["shape_index"] = (placeholder["crown_perimeter"]**2) / placeholder["crown_projection_area"]
        placeholder["form_factor"] = placeholder["crown_projection_area"] / (placeholder["crown_perimeter"]**2)
        # Useful Robust Features
            # Can remove compactness
        placeholder["compactness"] = (4 * math.pi * placeholder["crown_projection_area"]) / (placeholder["crown_perimeter"]**2)
        placeholder["convexity"] = placeholder["crown_perimeter"] / convex_perimeter
        placeholder["solidity"] = placeholder["crown_projection_area"] / placeholder["geometry"].convex_hull.area # convex hull score
        placeholder["elongation"] = placeholder["major_axis"] / placeholder["minor_axis"]

        # TODO: Add in Bending energy, first and second order invariant moment
        placeholder["bendingE"] = list(map(engineer._bendingEnergy, placeholder["geometry"], placeholder["radius_of_gyration"]))
        
        # calling upon a global variable.
    
        # Removing the bottom features makes detection slightly worse. So will keep them
        # for EDA and decide from there.
        # # Drop useless/repeat/non-robust items
        #placeholder.drop(["circularity", "form_factor", "minor_axis", "major_axis", "radius_of_gyration", "crown_perimeter", "crown_projection_area"], axis=1, inplace = True)
        #placeholder.drop(["minor_axis", "major_axis", "radius_of_gyration", "crown_perimeter", "crown_projection_area"], axis=1, inplace = True)
        return placeholder

    # https://gis.stackexchange.com/questions/297076/how-to-calculate-mean-value-of-a-raster-for-each-polygon-in-a-shapefile
    @staticmethod
    def _CoV(x):
        # Coefficient of Determination
        return np.ma.std(x) / np.ma.mean(x) * 100

    @staticmethod
    def _detStats(x, geometry):
        geometry = geometry.to_crs(x.rio.crs)
        #nodata = x.rio.nodata
        affine = x.rio.transform()
        array = np.array(x)[0]
        # return pd.DataFrame(zonal_stats(geometry, array, 
        #             affine=affine, 
        #             stats="max majority mean",
        #             add_stats={"CoV":CoV}))
        return pd.DataFrame(zonal_stats(geometry, array, nodata=np.nan,
                    affine=affine,
                    stats="mean"))#,
                    #add_stats={"CoV":engineer._CoV}))   
        

    # TODO: Select a better statistic mean/median/mode/max/etc.
    # TODO: Speed up zonal statistics
    def zonalStatistics(self, placeholder, spectral):
        # Spectral Features
        geom = placeholder.loc[:,"geometry"]

        placeholder[["DEM_mean"]] = engineer._detStats(spectral["dem"], geom)
        placeholder[["NIR_mean"]] = engineer._detStats(spectral["nir"], geom)
        placeholder[["Red_mean"]] = engineer._detStats(spectral["red"], geom)
        placeholder[["Reg_mean"]] = engineer._detStats(spectral["reg"], geom)
        placeholder[["NDRE_mean"]] = engineer._detStats(spectral["ndre"], geom)
        placeholder[["NDVI_mean"]] = engineer._detStats(spectral["ndvi"], geom)
        placeholder[["GNVDI_mean"]] = engineer._detStats(spectral["gndvi"], geom)
        placeholder[["ENDVI_mean"]] = engineer._detStats(spectral["endvi"], geom)
        placeholder[["Intensity_mean"]] = engineer._detStats(spectral["intensity"], geom)
        placeholder[["Saturation_mean"]] = engineer._detStats(spectral["saturation"], geom)
        return placeholder

    # TODO: https://iopscience.iop.org/article/10.1088/1361-6560/abfbf5/data
    def distanceFeatures(self, placeholder):
        # Distance Based Features
        # the reason that it is k + 1 is because the first neighbour is the point itself
        k = 4
        neigh = KNN(n_neighbors= k + 1)
        neigh.fit(placeholder[["latitude","longitude"]])
        distances, positions = neigh.kneighbors(placeholder[["latitude","longitude"]], return_distance=True)
        placeholder[["dist1", "dist2", "dist3", "dist4"]] = distances[:,1:]

        placeholder = placeholder.to_crs(4326)
        return placeholder

    @staticmethod
    def _scaleData(x):
        """
        
        """
        scaler = RobustScaler()
        # scaler = StandardScaler()
        return(scaler.fit_transform(x))


    def scaleData(self, placeholder, spectral, scale):
        placeholder["centroid"] = shapely.centroid(placeholder.loc[:,"geometry"])
        placeholder["latitude"] = placeholder["centroid"].y
        placeholder["longitude"] = placeholder["centroid"].x
        # put the spatial features first
        placeholder = placeholder.loc[:, ['geometry', 'centroid', 'latitude', 'longitude', 'confidence']]
        placeholder = self.shapeDescriptors(placeholder)
        # placeholder = self.distanceFeatures(placeholder)
        placeholder = placeholder.to_crs(4326)
        # zonal statistics have to come last as crs is 4326 
        # and above crs is converted to 3857 to work with the shape descriptors
        placeholder = self.zonalStatistics(placeholder, spectral)

        # print(self.spectralData['rgb'])
        analysis = ["z" + str(x) for x in range(25)] + ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', "dissimilarity"]
        placeholder[analysis] = placeholder["geometry"].apply(lambda x: self.imageA(x, spectral['rgb']))
        #placeholder[['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', "dissimilarity"]] = placeholder["geometry"].apply(lambda x: self._texture(x, spectral['rgb']))

        # placeholder.drop(['dist1', 'dist2', 'dist3', 'dist4'], axis = 1, inplace=True)
        # Feature Scaling
        # TODO: this paper says to use robust scaling: https://link.springer.com/article/10.1007/s00138-023-01450-x#Sec3
        #       For some reason it does work better than standard scaling
        if (scale):
            placeholder.loc[:, "confidence":] = engineer._scaleData(placeholder.loc[:,'confidence':])
        
        self.data = placeholder
        self.delineations = placeholder[["geometry"]]
        #return placeholder