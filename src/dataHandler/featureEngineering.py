# %%

import shapely.plotting
import shapely
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from osgeo import ogr, gdal
from osgeo import gdalconst
from rasterio.mask import mask
import earthpy.spatial as es
import earthpy.plot as ep
import earthpy as et
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


from sklearn.neighbors import NearestNeighbors as KNN
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
# parent class
from dataHandler.dataCollect import collect
from sklearn.preprocessing import StandardScaler

from rasterstats import zonal_stats

class engineer(collect):

    def __init__(self, num, tifs, geojsons, zips):
        super().__init__(num, tifs, geojsons, zips)
        self.scaleData(self.delineations, self.spectralData)
        

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

    # TODO: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
    # TODO: https://iopscience.iop.org/article/10.1088/1361-6560/abfbf5/data
    def shapeDescriptors(self, placeholder):
        placeholder =  placeholder.to_crs(3857)
        # Shape Descriptors Not Robust
        placeholder["crown_projection_area"] = shapely.area(placeholder.loc[:,"geometry"])
        placeholder["crown_perimeter"] = shapely.length(placeholder.loc[:,"geometry"])
        placeholder["radius_of_gyration"] = placeholder[["centroid", "geometry"]].apply(lambda x: engineer._radiusOfGyration(x.iloc[0], x.iloc[1].exterior.coords.xy[0], x.iloc[1].exterior.coords.xy[1]), axis=1)
        # Robust Shape Descriptors
        placeholder[["minor_axis", "major_axis"]] = placeholder["geometry"].apply(lambda x: engineer._major_minor(x))
        placeholder["isoperimetric"] = (4 * math.pi * placeholder["crown_projection_area"]) / (placeholder["crown_perimeter"]**2)
        placeholder["shape_index"] = (placeholder["crown_perimeter"]**2) / placeholder["crown_projection_area"]
        placeholder["form_factor"] = placeholder["crown_projection_area"] / (placeholder["crown_perimeter"]**2)
        placeholder["circularity"] = (placeholder["crown_perimeter"]**2) / (4 * math.pi * placeholder["crown_projection_area"])
        placeholder["convexity"] = placeholder["crown_perimeter"] / placeholder["geometry"].convex_hull.length
        placeholder["solidity"] = placeholder["crown_projection_area"] / placeholder["geometry"].convex_hull.area # convex hull score
        placeholder["elongation"] = placeholder["major_axis"] / placeholder["minor_axis"]
        placeholder["roundness"] = (4 * placeholder["crown_projection_area"]) / (math.pi * placeholder["major_axis"]**2)
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
                    stats="mean",
                    add_stats={"CoV":engineer._CoV}))   
        

    # TODO: Select a better statistic mean/median/mode/max/etc.
    # TODO: Speed up zonal statistics
    def zonalStatistics(self, placeholder, spectral):
        # Spectral Features
        geom = placeholder.loc[:,"geometry"]

        placeholder[["DEM_mean", "DEM_CV"]] = engineer._detStats(spectral["dem"], geom)
        placeholder[["NIR_mean", "NIR_CV"]] = engineer._detStats(spectral["nir"], geom)
        placeholder[["Red_mean", "Red_CV"]] = engineer._detStats(spectral["red"], geom)
        placeholder[["Reg_mean", "Reg_CV"]] = engineer._detStats(spectral["reg"], geom)
        placeholder[["NDRE_mean", "NDRE_CV"]] = engineer._detStats(spectral["ndre"], geom)
        placeholder[["NDVI_mean", "NDVI_CV"]] = engineer._detStats(spectral["ndvi"], geom)
        placeholder[["GNVDI_mean", "GNVDI_CV"]] = engineer._detStats(spectral["gndvi"], geom)
        placeholder[["ENDVI_mean", "ENDVI_CV"]] = engineer._detStats(spectral["endvi"], geom)
        placeholder[["Intensity_mean", "Intensity_CV"]] = engineer._detStats(spectral["intensity"], geom)
        placeholder[["Saturation_mean", "Saturation_CV"]] = engineer._detStats(spectral["saturation"], geom)

        # More options if Zonal Statistics stops working
        # # https://github.com/shakasom/zonalstatistics/blob/master/Zonal_Statistics_Sentinel.ipynb
        # # https://corteva.github.io/geocube/html/examples/zonal_statistics.html
        # grouped_elevation = spectral["dem"].drop("spatial_ref").groupby(geom)
        # grid_mean = grouped_elevation.mean().rename({"dem": "elevation_mean"})
        # print(grid_mean)

        # placeholder[["Green_mean", "Green_CV"]] = engineer._detStats(spectral["green"], geom)
        # placeholder[["Blue_mean", "Blue_CV"]] = engineer._detStats(spectral["blue"], geom)
        # placeholder[["DEM_max", "DEM_majority", "DEM_mean", "DEM_CV"]] = engineer._detStats(spectral["dem"], geom)
        # placeholder[["NIR_max", "NIR_majority", "NIR_mean", "NIR_CV"]] = engineer._detStats(spectral["nir"], geom)
        # placeholder[["Red_max", "Red_majority", "Red_mean", "Red_CV"]] = engineer._detStats(spectral["red"], geom)
        # placeholder[["Reg_max", "Reg_majority", "Reg_mean", "Reg_CV"]] = engineer._detStats(spectral["reg"], geom)
        # placeholder[["Green_max", "Green_majority", "Green_mean", "Green_CV"]] = engineer._detStats(spectral["green"], geom)
        # placeholder[["Blue_max", "Blue_majority", "Blue_mean", "Blue_CV"]] = engineer._detStats(spectral["blue"], geom)
        # placeholder[["NDRE_max", "NDRE_majority", "NDRE_mean", "NDRE_CV"]] = engineer._detStats(spectral["ndre"], geom)
        # placeholder[["NDVI_max", "NDVI_majority", "NDVI_mean", "NDVI_CV"]] = engineer._detStats(spectral["ndvi"], geom)
        # placeholder[["GNVDI_max", "GNVDI_majority", "GNVDI_mean", "GNVDI_CV"]] = engineer._detStats(spectral["gndvi"], geom)
        # placeholder[["ENDVI_max", "ENDVI_majority", "ENDVI_mean", "ENDVI_CV"]] = engineer._detStats(spectral["endvi"], geom)
        # placeholder[["Intensity_max", "Intensity_majority", "Intensity_mean", "Intensity_CV"]] = engineer._detStats(spectral["intensity"], geom)
        # placeholder[["Saturation_max", "Saturation_majority", "Saturation_mean", "Saturation_CV"]] = engineer._detStats(spectral["saturation"], geom)
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

    def scaleData(self, placeholder, spectral):
        placeholder["centroid"] = shapely.centroid(placeholder.loc[:,"geometry"])
        placeholder["latitude"] = placeholder["centroid"].y
        placeholder["longitude"] = placeholder["centroid"].x
        # put the spatial features first
        placeholder = placeholder.loc[:, ['geometry', 'centroid', 'latitude', 'longitude', 'confidence']]
        placeholder = self.shapeDescriptors(placeholder)
        placeholder = self.distanceFeatures(placeholder)
        # zonal statistics have to come last as crs is 4326 
        # and above crs is converted to 3857 to work with the shape descriptors
        placeholder = self.zonalStatistics(placeholder, spectral)
        # Feature Scaling
        scaler = StandardScaler()
        placeholder.loc[:, "confidence":] = scaler.fit_transform(placeholder.loc[:,'confidence':])
        
        self.data = placeholder
        self.delineations = placeholder["geometry"]
        #return placeholder