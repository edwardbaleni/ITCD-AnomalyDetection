# %%

    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
    # To read in data as raster stacks
    # cant use gdal unless we use conda
    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html#Working-with-Multi-Band-Raster

    # https://automating-gis-processes.github.io/CSC18/index.html 
    # https://autogis-site.readthedocs.io/en/latest/ 

# %%
import dataCollect # contains os, glob, random

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
    # Have to work in Conda for gdal to work
from osgeo import ogr, gdal
from osgeo import gdalconst
from rasterio.mask import mask

import earthpy.spatial as es
import earthpy.plot as ep
import earthpy as et

import matplotlib.pyplot as plt

import numpy as np

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

Greens = []
Blues = []
    # To check description of raster
    # raster.descriptions
    # range is the sample size
for i in range(sampleSize):
    NIRs.append(rio.open([j for j in data_paths_tif[0] if "nir_native" in j][0]).read())
    Reds.append(rio.open([j for j in data_paths_tif[0] if "red_native" in j][0]))
    Regs.append(rio.open([j for j in data_paths_tif[0] if "reg_native" in j][0]))
    RGBs.append(rio.open([j for j in data_paths_tif[0] if "visible_5cm" in j][0], "w"))
    # Here we read in the DEM and RGB, change the meta data and save
    # new instance to memory, and close original file
     
    DEMs.append(rio.open([j for j in data_paths_tif[0] if "dem_native" in j][0]))
    # Red, Green, Blue can be taken from RGB
    # Red is the first, Green is second and Blue is third
    # You can see this when you change around the plotting
    # the colouring will change depending on how you order it
    # this makes me believe that the correct order of RGB must be kept
    # e.g. "show(RGBs[0].read([1,2,3]))"
    # vs   "show(RGBs[0].read([2,1,3]))"
    Greens.append(RGBs[i].read(2))
    Blues.append(RGBs[i].read(3))
    Points.append(gpd.GeoDataFrame.from_file(data_paths_geojson[i]))

    # es._stack_bands([Reds[0], NIRs[0]])

# Make function to change transform to same as bands for everything




# %%

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


# %%

def 
    # https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry

from shapely.ops import unary_union
from shapely import LineString, Point, Polygon, BufferCapStyle, BufferJoinStyle
from shapely import buffer
from shapely import normalize, Polygon
import shapely


# enclose this in a function

    # Obtain example file
a = gpd.GeoDataFrame.from_file(data_paths_geojson[1])
point = list(a.iloc[:,1])

    # Combine all intersecting polygons
point = unary_union(point)
    # Combine all non-intersecting polygons
boundary = shapely.coverage_union_all(point)
    # Get the convex hull of the entire area (different from concave hull)
    # concave hull will get all the indents
    # will get more specific boundary with concave
    # maybe go with concave
    # Concave actually doesn't work well at all
    # However, convex is not perfect.
    # Concave works when you change the ratio
boundary_periph = shapely.concave_hull(boundary, ratio=0.15)#shapely.convex_hull(boundary)
#poly = gpd.GeoSeries(boundary_periph, crs=RGBs[0].crs.data)
poly = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary_periph), crs=RGBs[0].crs.data)

#boundary = gpd.GeoSeries(unary_union(point))
boundary_periph
boundary_periph.boundary


# in a try - except, try mask , except create mask, and apply mask. 

# %%
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]
# %%

coords = getFeatures(poly)#shapely.to_geojson(boundary_periph, indent=1)#getFeatures(shapely.to_json(boundary_periph))
out_img, out_transform = mask(RGBs[1], shapes=coords, crop=True, all_touched=True, pad = True)
out_meta = RGBs[1].meta.copy()
print(out_meta)
epsg_code = int(RGBs[1].crs.data['init'][5:])
print(epsg_code)

out_meta.update({"driver": "GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform,
                 "crs": epsg_code})

out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out.tif"
with rio.open(out_tif, "w", **out_meta) as dest:
   dest.write(out_img)

# %%
clipped = rio.open(out_tif)
show((clipped), cmap='terrain')

# %%
    # We need a conventional way to save the data after reading it in so that 
    # Algorithms can handle it, be it image based methods or 
    # geometric methods
# https://medium.com/abraia/hyperspectral-image-classification-with-python-7dce4ebcda0a
# https://shreshai.blogspot.com/2014/11/converting-raster-dataset-to-xyz-in.html
# https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1
# https://github.com/IamShubhamGupto/BandNet/tree/master/notebooks

# %%
    # For image manipulation
# https://image-slicer.readthedocs.io/en/latest/functions.html
# https://www.youtube.com/watch?v=hNFNVmh1Qfs
# https://readthedocs.org/projects/image-slicer/downloads/pdf/latest/



# %% 

    # To save as np array is possible
    # https://github.com/IamShubhamGupto/BandNet/blob/master/notebooks/1_image_to_numpy.ipynb