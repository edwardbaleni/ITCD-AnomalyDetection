# %%
import dataCollect # contains os, glob, random

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
#from osgeo import ogr, gdal, gdalconst
from rasterio.mask import mask

import earthpy.spatial as es
import earthpy.plot as ep
import earthpy as et

import matplotlib.pyplot as plt

# %%
    # Collect file paths
data_paths_tif = dataCollect.dCollect(size=20, file_type="tif")
data_paths_geojson = dataCollect.dCollect(size=20, file_type="geojson")

    # Create raster stack in 


# %%

    # Open filelist and stack within erf
    # so that

ds = rio.open(data_paths_tif[2][4])

# For the geojsons, need to look whether unzipped folder
#   Only works for jsons
c = gpd.read_file(data_paths_geojson[2])


CO_BD= gpd.GeoDataFrame.from_file(data_paths_geojson[2])
# Plot them
fig, ax = plt.subplots(figsize=(5, 15))
rio.plot.show(ds, ax=ax)
CO_BD.plot(ax=ax, facecolor='none', edgecolor='blue')

# %%
# To read in data as raster stacks
# https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html#Working-with-Multi-Band-Raster

data = []






# %%

for i in path:
    filelist = os.listdir(i)
    #i.extractall()
    for j in filelist:
        print(j)
        if j.endswith(".tif"):
            file = rasterio.open(i + j)
            print(i + j)
            show(file)
        # elif j.endswith(".gz"):
        #     file_name = i + j
        #     shutil.unpack_archive(file_name)
        #if j.endswith(".geojson"):
        else:
             file2 = gpd.read_file(i + j)
             file2.plot()
