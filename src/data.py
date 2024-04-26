# %%
    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
import dataCollect # contains os, glob, random

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
    # Have to work in Conda for gdal to work
from osgeo import ogr, gdal
from osgeo import gdalconst
from rasterio.mask import mask

import pycrs

import earthpy.spatial as es
import earthpy.plot as ep
import earthpy as et

import matplotlib.pyplot as plt

# %%
    # Collect file paths
sampleSize = 20
data_paths_tif = dataCollect.dCollect(size=sampleSize, file_type="tif")
data_paths_geojson = dataCollect.dCollect(size=sampleSize, file_type="geojson")

    # Create raster stack in 


# %%

    # Open filelist and stack within erf
    # so that
DEMs = []
NIRs = []
Reds = []
Regs = []
RGBs = []
Points = []

    # range is the sample size
for i in range(sampleSize):
    DEMs.append(rio.open(data_paths_tif[i][0]))
    NIRs.append(rio.open(data_paths_tif[i][1]))
    Reds.append(rio.open(data_paths_tif[i][2]))
    Regs.append(rio.open(data_paths_tif[i][3]))
    RGBs.append(rio.open(data_paths_tif[i][4]))
    Points.append(gpd.GeoDataFrame.from_file(data_paths_geojson[i]))


# ds = rio.open(data_paths_tif[1][4])

# # For the geojsons, need to look whether unzipped folder
# #   Only works for jsons
# c = gpd.read_file(data_paths_geojson[1])


# CO_BD= gpd.GeoDataFrame.from_file(data_paths_geojson[1])
# # Plot them
# fig, ax = plt.subplots(figsize=(5, 15))
# rio.plot.show(ds, ax=ax)
# CO_BD.plot(ax=ax, facecolor='none', edgecolor='blue')


# %%

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


# %%

# from matplotlib import pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Polygon

# poly1 = Polygon([(0,0), (2,0), (2,2), (0,2)])
# poly2 = Polygon([(2,2), (4,2), (4,4), (2,4)])
# poly3 = Polygon([(1,1), (3,1), (3,3), (1,3)])
# poly4 = Polygon([(3,3), (5,3), (5,5), (3,5)])
# polys = [poly1, poly2, poly3, poly4] 

# gpd.GeoSeries(polys).boundary.plot()
# plt.show()

from shapely.ops import unary_union
from shapely import LineString, Point, Polygon, BufferCapStyle, BufferJoinStyle
from shapely import buffer
from shapely import normalize, Polygon
import shapely

# mergedPolys = unary_union(polys)

# gpd.GeoSeries([mergedPolys]).boundary.plot()
# plt.show()
a = gpd.GeoDataFrame.from_file(data_paths_geojson[0])
point = list(a.iloc[:,1])

# for i in range(len(point)):
#     point[i] = buffer(geometry = point[i], distance=5)
#     point[i]

point = unary_union(point)
boundary = shapely.coverage_union_all(point)
boundary = shapely.convex_hull(boundary)
#boundary = gpd.GeoSeries(shapely.coverage_union_all(list(point)))
#boundary = gpd.GeoSeries(unary_union(point))
boundary
boundary.boundary


# %%
# To read in data as raster stacks
# cant use gdal unless we use conda
# https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html#Working-with-Multi-Band-Raster
a["hold"] = 0
nepal_zone = a[['hold', 'geometry']]

zones = nepal_zone.dissolve(by='hold')
zones.boundary.plot()