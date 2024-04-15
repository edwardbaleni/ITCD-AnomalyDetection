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

# %%

os.chdir(data_paths_tif[0][0].index("\\", -1, 0))
multi_band = data_paths_tif[0]
multi_band

os.getcwd()
landsat_multi_path