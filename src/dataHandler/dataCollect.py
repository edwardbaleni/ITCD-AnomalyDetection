# %%
import os
import random
from glob import glob
import gzip
import geopandas as gpd
import xarray
import rioxarray as rxr
from rasterio.enums import Resampling

# This init is just to handle unzipping geojsons
# We can make it in terms of just geojsons
def dCollect(size, file_type):
    # For now unzip all jsons
    # From current directory get 
    os.chdir('Data/')
    pop_erf = os.listdir()

    # for now obtain a small subset of data from list of 316 files to test
    # Try 20 folders
    if size < 316:
        # obtain a random sample
        random.seed(2024)
        sample_erf = random.sample(pop_erf, size)
    else:
        sample_erf = pop_erf

    path_holder = []
    for files in sample_erf:
        path = []
        os.chdir(files + "\\")
        # Read in files
        path = glob(os.getcwd() + "/*." + file_type )
        path_holder.append(path)    
        # Move back to Data Directory
        os.chdir("..")    
    
    # Move back to src directory
    os.chdir("..")

    return path_holder

    # Collect file paths
# for trial implementation
# for final implementation, need to ask user to input file paths of 
# interest
# TODO: https://www.youtube.com/watch?v=YTOUBGHEgZg
# TODO: https://www.merge.dev/blog/get-folders-google-drive-api 

def collectFiles(sampleSize):
    # Collect file paths
    # for trial implementation
    # for final implementation, need to ask user to input file paths of 
    # interest
    sampleSize = 20
    data_paths_tif = dCollect(size=sampleSize, file_type="tif")
    data_paths_geojson = dCollect(size=sampleSize, file_type="geojson")
    data_paths_geojson_zipped = dCollect(size=sampleSize, file_type="gz")
    random.seed(2024)
    # Create raster stack in 
    # this is a safe way to open zipped files without extracting them
    # import gzip
    # file = "C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/93001/orchard_validation.geojson.gz"
    # with gzip.open(file, 'rb') as f:
    #     trys = gpd.read_file(f)
    return data_paths_tif, data_paths_geojson, data_paths_geojson_zipped

def retrieveData(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
    nir = xarray.open_dataarray([j for j in data_paths_tif if "nir_native" in j][0])
    red = xarray.open_dataarray([j for j in data_paths_tif if "red_native" in j][0])
    reg = xarray.open_dataarray([j for j in data_paths_tif if "reg_native" in j][0])
    rgb = xarray.open_dataarray([j for j in data_paths_tif if "visible_5cm" in j][0])
    dem = xarray.open_dataarray([j for j in data_paths_tif if "dem_native" in j][0])
    mask = gpd.read_file([j for j in data_paths_geojson if "survey_polygon" in j][0])
    with gzip.open([j for j in data_paths_geojson_zipped if "mask_rcnn.geojson" in j][0], 'rb') as f:
        points = gpd.read_file(f)
    with gzip.open([j for j in data_paths_geojson_zipped if "orchard_validation.geojson" in j][0], 'rb') as k:
        ref_Data = gpd.read_file(k)
    
    return dem, nir, red, reg, rgb, points, mask, ref_Data

# remove points that aren't in mask
def removePoints(geom, mask):
    # find points within mask
    return mask.contains(geom)

def recursivePointRemoval(geoms, mask):
    # remove points that touch the mask
    hold = []
    for i in range(0, len(geoms)):
        if removePoints(geoms.iloc[i,1], mask)[0]:
            hold.append(i)
    return hold

def repprojectData(data, xds_match):
        data = data.rio.reproject_match(xds_match, resampling = Resampling.nearest) # Resampling.bilinear
        data = data.assign_coords({
            "x": xds_match.x,
            "y": xds_match.y,})
        return data

# TODO: should be sped up from 20 seconds to less than 2 seconds
def fixData(tif, geojson, zipped):
    # Resamples data and removes irrelevant points
    dem, nir, red, reg, rgb, points, mask, ref_data = retrieveData(tif, geojson, zipped)

    # Need this to match the resolution of the data
            # print_raster(dem)
            # print_raster(nir)
            # print_raster(red)
            # print_raster(reg)
            # print_raster(rgb)
        # rgb has the highest resolution
        # nir most mid range resolution
        # dem has the lowest resolution 
        # To speed up the process, method
        # reproject using dem and nearest neighbour for fastest processing
        # then reproject using rgb and bilinear for standard processing
        # then reproject using rgb and gaussian for best processing
    xds_match = dem
    #xds_match = rgb

    # don't reproject xds_match, speed up x2
    # dem = repprojectData(dem, xds_match)
    nir = repprojectData(nir, xds_match)
    red = repprojectData(red, xds_match)
    reg = repprojectData(reg, xds_match)
    rgb = repprojectData(rgb, xds_match)
    green = rgb[1]
    blue = rgb[2]

    data = {"dem": dem, 
            "nir": nir, 
            "red": red, 
            "reg": reg, 
            "rgb": rgb, 
            "green": green, 
            "blue": blue}
    

    index_mask_intersect = recursivePointRemoval(points, mask)
    delineations = points.iloc[index_mask_intersect]

    return data, delineations, mask, ref_data

# TODO: select better vegetative indices 
def vegIndices(data):
        # reg stands for red edge
    
    data["ndvi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
    data["ndre"] = (data["nir"] - data["reg"]) / (data["nir"] + data["reg"])
    data["gndvi"] = (data["nir"] - data["Green"]) / (data["nir"] + data["Green"])
    data["endvi"] = ((data["nir"]+ data["Green"] - 2 * data["blue"]) / (data["nir"] + data["Green"] + 2 * data["blue"]))
    data["intensity"] = data["nir"] + data["Green"] + data["blue"]
    data["saturation"] = (data["Intensity"] -3 * data["blue"]) / data["Intensity"]
    #       as only NDVI seems to distinguish ground pixels from trees well
    data["NDVI"].plot()
    #data["NDRE"].plot()
    #data["GNDVI"].plot()
    #data["ENDVI"].plot()

    # data["NDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/NDVI.tif")
    # data["NDRE"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/NDRE.tif")
    # data["GNDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/GNDVI.tif")
    # data["ENDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/ENDVI.tif")
    # data["Intensity"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/Intensity.tif")
    # data["Satuartion"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/Saturation.tif")
    return data

# %%
from timeit import default_timer as timer
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # working directory is that where the file is placed
    #os.chdir("..")
    sampleSize = 20
    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = collectFiles(sampleSize)

    num = 0
    start = timer()
    data, delineations, mask, ref_data = fixData(data_paths_tif[num], data_paths_geojson[num], data_paths_geojson_zipped[num])
    data = vegIndices(data)
    end = timer()
    print(end - start)

    # Plotting
    mask = mask
    tryout = data["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    tryout = tryout/255
    fig, ax = plt.subplots(figsize=(15, 15))
    tryout.plot.imshow(ax=ax)
    delineations.plot(ax=ax, facecolor = 'none',edgecolor='red') 
