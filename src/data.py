# %%
import dataCollect

import glob

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from osgeo import ogr, gdal, gdalconst


import matplotlib as plt




# %%

data_paths = dataCollect.dCollect(size=20)

    # Create raster stack in 


# %%

    # Open filelist and stack within erf
    # so that

ds = rio.open(data_paths[0][0])

# For the geojsons, need to look whether unzipped folder
#   Only works for jsons
c = gdp.read_file(data_paths[0][1])


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
