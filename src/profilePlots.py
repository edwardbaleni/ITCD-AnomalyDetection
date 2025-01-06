import numpy as np
import rasterio as rio
import utils

import pandas as pd

import plotly.graph_objects as go

import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiprocessing import Pool

def getDataNames(sampleSize):
    return utils.collectFiles(sampleSize)

def process(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
    # Your code goes here
    # num = 0
    # Do not scale data here because it will be split into training and testing data
    myData = utils.engineer(0,
                            [data_paths_tif], 
                            [data_paths_geojson], 
                            [data_paths_geojson_zipped],
                            False) # False)
    
    mask = myData.mask
    spectralData = myData.spectralData
    
    return spectralData, mask

if __name__ == '__main__':
    # Get sample size from user
    sampleSize = 5

    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = getDataNames(sampleSize)

    # I have 20 cores!
    with Pool(5) as pool:
        args = zip(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
        results = pool.starmap(process, list(args))

    spectralD, masks = zip(*results)
    spectralD = list(spectralD)

    # Mask the spectral data using the mask
    masked_spectralD = []
    for spectral, mask in zip(spectralD, masks):
        masked = {}
        for key, value in spectral.items():
            masked[key] = value.rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
        masked_spectralD.append(masked)
    
    # Although we are plotting the whole thing. 
    # We only need a 3D of DEM as there may be an existent slope there
    for i, masked in enumerate(masked_spectralD):
        for key, value in masked.items():
            if key == "rgb":
                continue
            elif key == 'green' or key == 'blue':
                x = value.x.values
                y = value.y.values
                z = value.values
            else: 
                x = value.x.values
                y = value.y.values
                z = value.values[0]

            # Downscale the data to lower resolution
            scale_factor = 0.1  # Adjust this factor to change the resolution
            min_positive_z = np.min(z[z > 0])
            z[z < 0] = min_positive_z
            x_downscaled = x[::int(1/scale_factor)]
            y_downscaled = y[::int(1/scale_factor)]
            z_downscaled = z[::int(1/scale_factor), ::int(1/scale_factor)]

            # Create the figure with downscaled data
            fig = go.Figure(data=[go.Surface(z=z_downscaled, x=x_downscaled, y=y_downscaled, colorscale='Viridis')])
            fig.write_html(f"results/EDA/3D/Orchard_{i}_{key}.html")
    

    # Profile Plots
    # TODO: Label the colorbar somehow
    for i, masked in enumerate(masked_spectralD):
        for key, value in masked.items():
            if key == "rgb":
                continue
            fig, ax = plt.subplots(figsize=(20, 20))
            value.plot(ax=ax, cmap="viridis")
            ax.set_axis_off()
            ax.set_title("")
            plt.savefig(f"results/EDA/Profiles/Orchard_{i}_{key}.png")
            plt.close(fig)