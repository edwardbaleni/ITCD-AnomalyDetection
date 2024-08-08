# %%

    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
    # To read in data as raster stacks
    # cant use gdal unless we use conda
    # https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html#Working-with-Multi-Band-Raster

    # https://automating-gis-processes.github.io/CSC18/index.html 
    # https://autogis-site.readthedocs.io/en/latest/ 




# How about we make this a whole file a class
# that works on one file at a time
# then we use another python file to loop over all images or folders
# and in this way this part can be parellizable
# since the algorhtims are each rely solely on the one image alone.


# %%
import shapely.plotting
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
import pandas as pd
import math

from sklearn.neighbors import NearestNeighbors as KNN

import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator

import random

import rioxarray as rxr
import xarray
from rasterio.enums import Resampling

import gzip


# %%
    # Collect file paths
# for trial implementation
# for final implementation, need to ask user to input file paths of 
# interest
# TODO: https://www.youtube.com/watch?v=YTOUBGHEgZg
# TODO: https://www.merge.dev/blog/get-folders-google-drive-api 
sampleSize = 20
data_paths_tif = dataCollect.dCollect(size=sampleSize, file_type="tif")
data_paths_geojson = dataCollect.dCollect(size=sampleSize, file_type="geojson")
data_paths_geojson_zipped = dataCollect.dCollect(size=sampleSize, file_type="gz")
random.seed(2024)
# Create raster stack in 
# this is a safe way to open zipped files without extracting them
# import gzip
# file = "C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/93001/orchard_validation.geojson.gz"
# with gzip.open(file, 'rb') as f:
#     trys = gpd.read_file(f)


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
masks = []
Ref_Data = []

    # To check description of raster
    # raster.descriptions
    # range is the sample size
for i in range(sampleSize):
    NIRs.append(xarray.open_dataarray([j for j in data_paths_tif[i] if "nir_native" in j][0]))
    Reds.append(xarray.open_dataarray([j for j in data_paths_tif[i] if "red_native" in j][0]))
    Regs.append(xarray.open_dataarray([j for j in data_paths_tif[i] if "reg_native" in j][0]))
    RGBs.append(xarray.open_dataarray([j for j in data_paths_tif[i] if "visible_5cm" in j][0]))
    # Here we read in the DEM and RGB, change the meta data and save
    # new instance to memory, and close original file
     
    DEMs.append(xarray.open_dataarray([j for j in data_paths_tif[i] if "dem_native" in j][0]))
    # Red, Green, Blue can be taken from RGB
    # Red is the first, Green is second and Blue is third
    # You can see this when you change around the plotting
    # the colouring will change depending on how you order it
    # this makes me believe that the correct order of RGB must be kept
    # e.g. "show(RGBs[0].read([1,2,3]))"
    # vs   "show(RGBs[0].read([2,1,3]))"
    #Greens.append(RGBs[i].read(2))
    #Blues.append(RGBs[i].read(3))
    masks.append(gpd.read_file([j for j in data_paths_geojson[i] if "survey_polygon" in j][0]))
    with gzip.open([j for j in data_paths_geojson_zipped[i] if "mask_rcnn.geojson" in j][0], 'rb') as f:
        Points.append(gpd.read_file(f))
    with gzip.open([j for j in data_paths_geojson_zipped[i] if "orchard_validation.geojson" in j][0], 'rb') as k:
        Ref_Data.append(gpd.read_file(k))
#    Points.append(gpd.read_file(data_paths_geojson[i]))#GeoDataFrame.from_file(data_paths_geojson[i]))

    # es._stack_bands([Reds[0], NIRs[0]]) # to stack bands


# %%
# TODO: remove all points that touch the mask

# %%
# merge dataframes
# Make function to change transform to same as bands for everything

num = 19

xds_DEM = DEMs[num] #xarray.open_dataarray(data_paths_tif[ num ][0])
xds_NIR = xds_match = NIRs[num] #xds_match = xarray.open_dataarray(data_paths_tif[ num ][1])
xds_Red = Reds[num] #xarray.open_dataarray(data_paths_tif[ num ][2])
xds_Reg = Regs[num] #xarray.open_dataarray(data_paths_tif[ num ][3])
xds_RGB = RGBs[num] #xarray.open_dataarray(data_paths_tif[ num ][4])

xds_dictionary = {"DEM": xds_DEM, 
                  "NIR": xds_NIR, 
                  "Red": xds_Red, 
                  "Reg": xds_Reg, 
                  "RGB": xds_RGB}

# fig, axes = plt.subplots(ncols=2, figsize=(12,4))
# xds_dictionary["DEM"].plot(ax=axes[0])
# xds_dictionary["RGB"].plot(ax=axes[1])
# plt.draw()

data = {}
# xds_DEM_match = xds_DEM.rio.reproject_match(xds_match, resampling = Resampling.bilinear)
# xds_Red_match = xds_Red.rio.reproject_match(xds_match, resampling = Resampling.bilinear)
# xds_Reg_match = xds_Reg.rio.reproject_match(xds_match, resampling = Resampling.bilinear)
# xds_RGB_match = xds_RGB.rio.reproject_match(xds_match, resampling = Resampling.bilinear)

for key in xds_dictionary:
    data[key] = xds_dictionary[key].rio.reproject_match(xds_match, resampling = Resampling.bilinear)
    data[key] = data[key].assign_coords({
        "x": xds_match.x,
        "y": xds_match.y,
    })


# xds_repr_match = xds_repr_match.assign_coords({
#     "x": xds_match.x,
#     "y": xds_match.y,
# })

data["Green"] = data["RGB"][1]
data["Blue"] = data["RGB"][2]

# now we are able to perform calculations between the two rasters now they are in the same
#  projection, resolution, and extents
# we can perform calculations between the two rasters now they are in the same
# will be helpful when calculating NDVI and others.
# Find other vegetation indices to calculate
#diff = xds_repr_match - xds_match

# reg stands for red edge
data["NDVI"] = (data["NIR"] - data["Red"]) / (data["NIR"] + data["Red"])
data["NDRE"] = (data["NIR"] - data["Reg"]) / (data["NIR"] + data["Reg"])
data["GNDVI"] = (data["NIR"] - data["Green"]) / (data["NIR"] + data["Green"])
data["ENDVI"] = ((data["NIR"]+ data["Green"] - 2 * data["Blue"]) / (data["NIR"] + data["Green"] + 2 * data["Blue"]))
# TODO: add these to data table in background
data["Intensity"] = data["NIR"] + data["Green"] + data["Blue"]
data["Saturation"] = (data["Intensity"] -3 * data["Blue"]) / data["Intensity"]

# TODO: select better vegetative indices 
#       as only NDVI seems to distinguish ground pixels from trees well
data["NDVI"].plot()
#data["NDRE"].plot()
#data["GNDVI"].plot()
#data["ENDVI"].plot()

# %%
    # For image manipulation
# https://image-slicer.readthedocs.io/en/latest/functions.html
# https://www.youtube.com/watch?v=hNFNVmh1Qfs
# https://readthedocs.org/projects/image-slicer/downloads/pdf/latest/




# %%
# mask in gdal and plot
    # only need the masksing to illustrate anomalies. 
    # So we can either call our own mask provided by 
    # user or we can create our own mask using the algorithm made
    # either way, need to use rioxaray instead of straight rasterio
    # as we cannot assume user is willing to use disk space to
    # save the masked file.

# tryout
mask = masks[num]#gpd.read_file("C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/122075/survey_polygon.geojson")
tryout = xds_RGB[0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
Points[ num ].plot(ax=ax, facecolor = 'none',edgecolor='red') 

# fig, ax = plt.subplots(figsize=(15, 15))
# rio.plot.show(clipped, ax=ax)
# Points[ num ].plot(ax=ax, facecolor='none', edgecolor='blue')

# %%

                    # Feature Engineering

# %%
a = Points[ num ]
geom = a.iloc[:,1]


# %%

a["centroid"] = shapely.centroid(a.loc[:,"geometry"])

    # When the latitude and longitude of the centroids are included
    # we see that the anomaly scores aren't as high,
    # and that the anomalies are not as clear.
    # However, when these are included the false-positives are more apparent

    # the exculsion of the latitude and longitude of the centroids, or spatial features
    # identifies more anomalies both false-positives and a lot of segmentation issues
    # albeit I can not determine the accuracy of this method
a["latitude"] = a["centroid"].y
a["longitude"] = a["centroid"].x

# put the spatial features first
a = a.loc[:, ['geometry', 'centroid', 'latitude', 'longitude', 'confidence']]



# %%
                    # Shape Descriptors Not Robust
# %%
a["crown_projection_area"] = shapely.area(a.loc[:,"geometry"])
a["crown_perimeter"] = shapely.length(a.loc[:,"geometry"])

# https://www.tutorialspoint.com/radius-of-gyration
# above link is for radius of gyration

    # Wrong behaviours may be being learnt including absolute locations
    # they may not be a good idea to include.
    # Explanatory varaibles are more useful than absolute locations
  
# %%
# Radius of gyration - https://www.tutorialspoint.com/radius-of-gyration
# For now we will use the straight line distance from the centroid to the vertices
# However, in future this line should be robust, say in the case of a concave polygon
# the line should not exit the polygon.
# I want to calculate the distance between two points within a concave shape, 
# and the distance line between these points cannot exit the polygon.

# still need to find the sd for the radius of gyration
# https://www.researchgate.net/publication/6764643_Experimental_characterization_of_vibrated_granular_rings 

def radius(xx_centre, xx, yy):
    rad = 0
    short = 1000
    long = -1000
    # also save short and long radius to have cross-section
    # instead of lookinoutg at longest and shortest radius
    # it may be more helpful to look at the major and minor axis
    # http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
    for i in range(len(xx)):
        dist = xx_centre.distance(shapely.Point(xx[i], yy[i]))
        if dist < short:
            short = dist
        if dist > long:
            long = dist
        rad += dist**2
    # return radius of gyration, short radius, long radius
    # still need to output the standard deviation of the radius of gyration
    return pd.Series([math.sqrt(rad) / len(xx), short, long])


a[["radius_of_gyration", "short","long"]] = a[["centroid", "geometry"]].apply(lambda x: radius(x[0], x[1].exterior.coords.xy[0], x[1].exterior.coords.xy[1]), axis=1)



# %%
                    # Robust Shape Descriptors

# %%

# major and minor axis
# https://stackoverflow.com/questions/13536209/efficient-way-to-measure-region-properties-using-shapely
def major_minor(xx):
    mbr_points = list(zip(*xx.minimum_rotated_rectangle.exterior.xy))
    mbr_lengths = [shapely.LineString([mbr_points[i], mbr_points[i + 1]]).length for i in range(len(mbr_points) - 1)]
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)
    return pd.Series([minor_axis, major_axis])

a[["minor_axis", "major_axis"]] = a["geometry"].apply(lambda x: major_minor(x))

# %%
# Descriptors
    # compactness ratios
a["isoperimetric"] = (4 * math.pi * a["crown_projection_area"]) / (a["crown_perimeter"]**2)
a["shape_index"] = (a["crown_perimeter"]**2) / a["crown_projection_area"]
a["form_factor"] = a["crown_projection_area"] / (a["crown_perimeter"]**2)
a["circularity"] = (a["crown_perimeter"]**2) / (4 * math.pi * a["crown_projection_area"])
a["convexity"] = a["crown_perimeter"] / a["geometry"].convex_hull.length
a["solidity"] = a["crown_projection_area"] / a["geometry"].convex_hull.area # convex hull score
a["elongation"] = a["major_axis"] / a["minor_axis"]
a["roundness"] = (4 * a["crown_projection_area"]) / (math.pi * a["major_axis"]**2)




# %%

    # Need to remove radius, major and minor axis, crown projection area, crown perimeter,
    # radius of gyration, short, long
    # as they are not robust features
    # Small trees are still trees and should be identified as such

    # can remove confidence but don't necessarily have to as it is a feature
    # that we can expext most datasets to have. Aerobotics provides this feature
    # with every dataset.


# %%
                    # Vegetative Indices

# get data within geometry
# https://gis.stackexchange.com/questions/328128/extracting-data-within-geometry-shape/328320#328320
touch = xds_match.rio.clip([Points[ num ].iloc[0,1]], xds_match.rio.crs)
touch.plot()
# this is the mean of the raster within the polygon for one polygon. Weighted average will be a bit dificult to collect.
# we could also just obtain the max or the median instead or the mode instead of the mean or weighted average
print(touch.mean()) 


# %%
from rasterstats import zonal_stats
# https://gis.stackexchange.com/questions/297076/how-to-calculate-mean-value-of-a-raster-for-each-polygon-in-a-shapefile
# Code down from 30 minutes to 30 seconds
# Need to make sure that I am reprojecting correctly !!!
def CoV(x):
    # Coefficient of Determination
    return np.ma.std(x) / np.ma.mean(x) * 100

def detStats(data):
    affine = data.rio.transform()#
    array = data.to_numpy()[0]
    return pd.DataFrame(zonal_stats(a["geometry"], array, 
                affine=affine, 
                stats="max majority mean",
                add_stats={"CoV":CoV}))

a[["DEM_max", "DEM_majority", "DEM_mean", "DEM_CV"]] = detStats(data["DEM"])
a[["NDRE_max", "NDRE_majority", "NDRE_mean", "NDRE_CV"]] = detStats(data["NDRE"])
a[["NDVI_max", "NDVI_majority", "NDVI_mean", "NDVI_CV"]] = detStats(data["NDVI"])
a[["GNVDI_max", "GNVDI_majority", "GNVDI_mean", "GNVDI_CV"]] = detStats(data["GNDVI"])
a[["ENDVI_max", "ENDVI_majority", "ENDVI_mean", "ENDVI_CV"]] = detStats(data["ENDVI"])
a[["Intensity_max", "Intensity_majority", "Intensity_mean", "Intensity_CV"]] = detStats(data["Intensity"])

#affine = data["DEM"].rio.transform()#
#array1 = data["DEM"].to_numpy()[0]
#zonal_stats(a["geometry"], array1, 
#            affine=affine, 
#            stats="max majority",
#            add_stats={"mean":mymean})#, "max":mymax, "median":mymedian})


# %%

                    # Distance Based Features

# %%
# number of neighbours to find,
# the reason that it is k + 1 is because the first neighbour is the point itself
k = 4
neigh = KNN(n_neighbors= k + 1)
neigh.fit( a[["latitude","longitude"]])
distances, positions = neigh.kneighbors(a[["latitude","longitude"]], return_distance=True)
a[["dist1", "dist2", "dist3", "dist4"]] = distances[:,1:]





# %%

                    # Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
a.loc[:, "confidence":] = scaler.fit_transform(a.loc[:,'confidence':])

# %%    
                    # Feature Selection (if too many features)

# TODO: Feature selection

# %%
# from heatmap import corrplot

# plt.figure(figsize=(20, 20))
# corrplot(a.corr())

import seaborn as sns
# ax = sns.heatmap(a.corr(), annot=True)
# g = sns.PairGrid(a.loc[:, "confidence":])
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# g.add_legend()

sns.heatmap(a.loc[:, "confidence":].corr(), annot=False, cmap="crest")




# %%

                    # Histogram Method
# %%
    # Can simply look into outliers in the data here
fig = plt.figure(figsize =(10, 10))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(a.loc[:,"confidence":])
# show plot
plt.show()



# %%

                    # Extended Isolation Forest

# %%
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html#examples
# https://github.com/sahandha/eif/blob/master/Notebooks/TreeVisualization.ipynb 
# Set the predictors
h2o.init()
# %%
h2o_df = h2o.H2OFrame(a.loc[:,[ 'confidence',
       'crown_projection_area', 'crown_perimeter', 'radius_of_gyration',
       #'short', 'long', 
       'minor_axis', 'major_axis', 
       'isoperimetric', 'shape_index', 'form_factor', 'circularity',
       'convexity', 'solidity', 'elongation', 'roundness', 'DEM_max', 'NDRE_max',
       'NDVI_max', 'GNVDI_max', 'ENDVI_max', #'NDRE_median', 
       #'NDVI_median',
       #'GNVDI_median', 'ENDVI_median', 
       'dist1', 'dist2', 'dist3', 'dist4']])
predictors = [ 'confidence',
       'crown_projection_area', 'crown_perimeter', 'radius_of_gyration',
       #'short', 'long', 
       'minor_axis', 'major_axis', 
       'isoperimetric', 'shape_index', 'form_factor', 'circularity',
       'convexity', 'solidity', 'elongation', 'roundness', 'DEM_max', 'NDRE_max',
       'NDVI_max', 'GNVDI_max', 'ENDVI_max', #'NDRE_median', 
       #'NDVI_median',
       #'GNVDI_median', 'ENDVI_median', 
       'dist1', 'dist2', 'dist3', 'dist4']#list(a.columns)

# %%
# Extended Isolation Forest is a great unsupervised method for anomaly detection
# however, it does not allow for the use of spatial features

# Define an Extended Isolation forest model
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 1000,
                                          sample_size = int(len(a) * 0.8),
                                          extension_level = 6)#len(predictors) - 1)

# Train Extended Isolation Forest
eif.train(x = predictors,
          training_frame = h2o_df)

# Calculate score
eif_result = eif.predict(h2o_df)

# Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
# or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
anomaly_score = eif_result["anomaly_score"]

# Average path length  of the point in Isolation Trees from root to the leaf
mean_length = eif_result["mean_length"]

# %%
b = eif_result.as_data_frame()
# for Points[1]
# when the confidence variable is included then use a thresholf of 0.65
# when it is not included then use a threshold of 0.5, however, this picks out a lot more anomalies
# that may in fact be normal.
# It is likely that most papers will not include the confidence variable

# when scaled and no confidence then 0.5 is a good threshold, most of the anomalies are captured
# the same anomalies are captured with or withour using confidence anyway. So we may be able to get away with
# not using confidence
anomaly = a[b["anomaly_score"] > 0.5]
nominal = a[b["anomaly_score"] <= 0.5]

# %%

# fig, ax = plt.subplots(figsize=(20, 20))
# rio.plot.show(clipped, ax=ax)
# anomaly.plot(ax=ax, facecolor='none', edgecolor='red')
# nominal.plot(ax=ax, facecolor='none', edgecolor='blue')

#mask = gpd.read_file("C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/122075/survey_polygon.geojson")
#tryout = xds_RGB[0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
#tryout = tryout/255
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly.plot(ax=ax, facecolor='none', edgecolor='red')
nominal.plot(ax=ax, facecolor='none', edgecolor='blue')





# %%
from sklearn.neighbors import LocalOutlierFactor
# TODO: this is not the correct way to do this,
#       look into https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-novelty-detection-py

clf = LocalOutlierFactor(n_neighbors=20, novelty=True,contamination=0.087)
y_pred = clf.fit_predict( a.loc[:, 'confidence':])
#n_errors = (y_pred != ground_truth).sum()
#X_scores = clf.negative_outlier_factor_
anomaly_1 = a[y_pred < 0]
nominal_1 = a[y_pred > 0]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
anomaly_1.plot(ax=ax, facecolor='none', edgecolor='red')
nominal_1.plot(ax=ax, facecolor='none', edgecolor='blue')

# %%




            # do not do delauney triangulation with EFI use, nearest neighbour instead to get closest points 
            # otherwise the number of variables won't be contained at each vairable
            # Only use delauney triangulation for second method
            # Use distatnces to centres not to vertices

# %%
                    # Delauney Paper
# %%

# https://gis.stackexchange.com/questions/459091/definition-of-multipolygon-distance-in-shapely
import shapely.plotting

shapely.plotting.plot_polygon(a.iloc[0,1], color = "red")
shapely.plotting.plot_polygon(a.iloc[4,1], color = "blue")
plt.show()

# this distance outputs the distance from the nearest vertex to the nearest vertex of the
# polygons not from the centroid to the centroid
print("distance: ", {a.iloc[0,1].distance(a.iloc[4,1])})

# (-8.28502 - -8.28497) = 0.00005       # From closrset vertex to closest vertex
# (-8.28506 - -8.28495) = 0.00011       # From centroid to centroid\

# don't have to do polygons to polygons can do centroid to centtroid.

# %% 
from scipy.spatial import Delaunay
points = a.iloc[0:50,5:7].to_numpy()
tri = Delaunay(points)


# %%
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()

# try both delauney triangulation method and a nearest neighbour or could use a sort

# %%

# fig, ax = plt.subplots(figsize=(20, 20))
# rio.plot.show(clipped, ax=ax)
# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()
          


# %%

# because distances aren't already worked out from Delauney, we can
# do this manually from polygon to polygon instead of from vertex to vertex

# Simplices are the indices of the vertices that make up the triangle in
# points. If we match this to the centroid in main dataframe then we could 
# find distances between polygons.