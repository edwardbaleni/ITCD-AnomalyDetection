# %%
import dataHandler

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import shapely

# TODO: Speed up dataCollect

# %%
if __name__ == "__main__":
    # working directory is that where the file is placed
    # os.chdir("..")
    sampleSize = 20
    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = dataHandler.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
    num = 0
    myData = dataHandler.engineer(num, data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
    
    data = myData.data
    delineations = myData.delineations # = myData.data["geometry"]
    mask = myData.mask
    spectralData = myData.spectralData

    # Plotting
    mask = mask
    tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    tryout = tryout/255
    fig, ax = plt.subplots(figsize=(15, 15))
    tryout.plot.imshow(ax=ax)
    delineations.plot(ax=ax, facecolor = 'none',edgecolor='red') 

# %%
