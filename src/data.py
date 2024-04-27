# %%

    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
    # To read in data as raster stacks
    # cant use gdal unless we use conda
    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html#Working-with-Multi-Band-Raster

    # https://automating-gis-processes.github.io/CSC18/index.html 
    # https://autogis-site.readthedocs.io/en/latest/ 

# %%
import dataCollect # contains os, glob, random

import shapely # Used for mask creation

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
    NIRs.append(rio.open([j for j in data_paths_tif[i] if "nir_native" in j][0]))
    Reds.append(rio.open([j for j in data_paths_tif[i] if "red_native" in j][0]))
    Regs.append(rio.open([j for j in data_paths_tif[i] if "reg_native" in j][0]))
    RGBs.append(rio.open([j for j in data_paths_tif[i] if "visible_5cm" in j][0]))
    # Here we read in the DEM and RGB, change the meta data and save
    # new instance to memory, and close original file
     
    DEMs.append(rio.open([j for j in data_paths_tif[i] if "dem_native" in j][0]))
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

    # es._stack_bands([Reds[0], NIRs[0]]) # to stack bands

# Make function to change transform to same as bands for everything




# %%

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

# %%

def getMaskBounds(comp, shape):
    # https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry
    # enclose this in a function

        # Obtain example file
    # a = gpd.GeoDataFrame.from_file(data_paths_geojson[1])
    # point = list(a.iloc[:,1])
    point = list(shape.iloc[:,1])

        # Combine all intersecting polygons
    point = shapely.unary_union(point)
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
    poly = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary_periph), crs=comp.crs.data)

    return (poly)

# Masks data and gives all data the same transform
def getMask(rast, shape, placeHolder, out_tif="C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out.tif"):
    # Obtain masking, in terms of NIR file
    poly = getMaskBounds(comp=placeHolder, shape=shape)
    # Get features for masking
    coords = getFeatures(poly)
    # Get information to output masking
    out_img, out_transform = mask(rast, shapes=coords, crop=True, all_touched=True, pad = True)
    
    # The meta data has to be the same as 
    out_meta = placeHolder.meta.copy()
    # Change transform here
    out_transform = placeHolder.transform
    print(out_meta)
    epsg_code = int(rast.crs.data['init'][5:])
    print(epsg_code)

    out_meta.update({"driver": "GTiff",
                    "count": rast.count,
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": epsg_code})

    #out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out2.tif"
    with rio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)



# in a try - except, try mask , except create mask, and apply mask. 
# If we can't find masked file in data2 then mask it and obtain else obtain


# At this stage the masking is fine, 
# The transform is equal over all bands, however, the width and height isn't the same for each
# Everything except the width and height is good. Now we need to fix this
# Output all results to data2



# %%

getMask(NIRs[0], Points[0], NIRs[0],out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out1.tif")
getMask(DEMs[0], Points[0], NIRs[0],out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out2.tif")

# %%
clipped = rio.open(out_tif)
show((clipped), cmap='terrain')
#es._stack_bands([clipped, clipped1])

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