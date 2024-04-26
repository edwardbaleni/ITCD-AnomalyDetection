# %%
import dataCollect # contains os, glob, random

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from osgeo import ogr, gdal, gdalconst
from rasterio.mask import mask

import earthpy.spatial as es
import earthpy.plot as ep
import earthpy as et

import matplotlib.pyplot as plt

# %%
    # Collect file paths
data_paths_tif = dataCollect.dCollect(size=20, file_type="tif")

# %%
    # RGB files have different transform to other bands
    # will have RGB separate to bands
    # If we use geometric methods, then bands will be handy
    # If we use image methods then RGB will be very important

    # Dem also does not stack well. Will have to change the meta data to stack rgb and dem with 
    # bands. For now skip this step

    # This code creates multiband rasters
# multi_bands = data_paths_tif[1][1:4] 
# multi_path = data_paths_tif[1][0][0:data_paths_tif[0][0].rindex("\\") + 1] + "multi_band.tif"

# land_stack, land_meta = es.stack(multi_bands,
#                                  multi_path)

# %%
    # This code opens multiband raster
# with rio.open(multi_path) as src:
#     landsat_multi = src.read() 

# band_titles = [ "NIR", "RED", "REG"]
# ep.plot_bands(landsat_multi,
#               title=band_titles, cbar=False)
# plt.show()

# %%
data_paths_geojson = dataCollect.dCollect(size=20, file_type="geojson")
DEM = rio.open(data_paths_tif[1][0])
RGB = rio.open(data_paths_tif[1][5])
Bands = rio.open(multi_path) 
Points = gpd.read_file(data_paths_geojson[1])

# %%























# %%
# # need to make all dtype float32
# give everything the same width and height
# 

# band_titles = ["DEM", "NIR", "RED", "REG"]
# ep.plot_bands(landsat_multi,
#               title=band_titles, cbar=False)
# plt.show()

# %%
# file = data_paths_tif[0][0][data_paths_tif[0][0].rindex("\\") : 126] #: data_paths_tif[0][0].rindex("_")]
# file = data_paths_tif[0][0][0:data_paths_tif[0][0].rindex("\\")]
# data_paths_tif[0][0][104 + 1 : 111]
multi_bands = data_paths_tif[0][0:5] 
#multi_path = data_paths_tif[0][0][0:data_paths_tif[0][0].rindex("\\") + 1] + "multi_band.tif"

# %%
land_stack, land_meta = es.stack(multi_bands,
                                 multi_path)

# %%

# for i in range(len(multi_bands)):
with rio.open(multi_bands[1]) as src:
    new_meta = src.meta.copy()
