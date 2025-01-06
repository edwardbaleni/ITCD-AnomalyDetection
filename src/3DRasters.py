# %%

# plot profile plots of Red, Green, Blue, NIR, RedEdge, DEM
# plot these in 3D space individually for each of the 5 orchards of interest
# https://gis.stackexchange.com/questions/66367/display-a-georeferenced-dem-surface-in-3d-matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import rasterio as rio
import utils

def getDataNames(sampleSize):
    return utils.collectFiles(sampleSize)

sampleSize = 5

data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = getDataNames(sampleSize)

# %%
# read in the DEM

dem = rio.open(data_paths_tif[0][0])

# Read the DEM data
dem_data = dem.read(1)

# Get the coordinates
x, y = np.meshgrid(np.arange(dem_data.shape[1]), np.arange(dem_data.shape[0]))
z = dem_data





# %%

from osgeo import gdal
import numpy as np

# Open the raster file
file_path = data_paths_tif[0][0]  # Replace with your DEM file path
dataset = gdal.Open(file_path)


# %%

# Get geotransform and raster band
geotransform = dataset.GetGeoTransform()
band = dataset.GetRasterBand(1)

# Read raster data as a NumPy array
z = band.ReadAsArray()

# Extract geotransform parameters
x_origin = geotransform[0]
y_origin = geotransform[3]
pixel_width = geotransform[1]
pixel_height = geotransform[5]

# Create x and y coordinate arrays
rows, cols = z.shape
x = x_origin + np.arange(cols) * pixel_width
y = y_origin + np.arange(rows) * pixel_height

# Create a meshgrid of coordinates
x_grid, y_grid = np.meshgrid(x, y)

# Flatten arrays for x, y, and z coordinates
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()
z_flat = z.flatten()

# Print the first few coordinates (x, y, z)
for i in range(10):  # Change this number as needed
    print(f"x: {x_flat[i]}, y: {y_flat[i]}, z: {z_flat[i]}")
# %%

from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(x,y,z_flat,c=z_flat,cmap=plt.cm.jet)  
plt.savefig("random.png")

# %%
import numpy as np
from matplotlib.mlab import griddata
# craation of a 2D grid
xi = np.linspace(min(x_flat), max(x_flat))
yi = np.linspace(min(y_flat), max(y_flat))
X, Y = np.meshgrid(xi, yi)
# interpolation
Z = griddata(x, y, z, xi, yi)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=1, antialiased=True)
plt.show()