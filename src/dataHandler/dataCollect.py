import os
import random
from glob import glob
import gzip
import geopandas as gpd
import xarray
from rasterio.enums import Resampling
#import pyogrio

class collect:
    def __init__(self, num, tifs, geojsons, zips):
        self._num = num
        self._tif, self._geojson, self._zipped = tifs[self._num], geojsons[self._num], zips[self._num]
        self.spectralData, self.delineations, self.mask, self.ref_data = self.fixData(self._tif, self._geojson, self._zipped)      
        self._vegIndices(self.spectralData)  

    @staticmethod
    def _retrieveData(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
        nir = xarray.open_dataarray([j for j in data_paths_tif if "nir_native" in j][0])
        red = xarray.open_dataarray([j for j in data_paths_tif if "red_native" in j][0])
        reg = xarray.open_dataarray([j for j in data_paths_tif if "reg_native" in j][0])
        rgb = xarray.open_dataarray([j for j in data_paths_tif if "visible_5cm" in j][0])
        dem = xarray.open_dataarray([j for j in data_paths_tif if "dem_native" in j][0])
        mask = gpd.read_file([j for j in data_paths_geojson if "survey_polygon" in j][0], engine='pyogrio',use_arrow=True)
        with gzip.open([j for j in data_paths_geojson_zipped if "mask_rcnn.geojson" in j][0], 'rb') as f:
            points = gpd.read_file(f, engine='pyogrio', use_arrow=True)
        with gzip.open([j for j in data_paths_geojson_zipped if "orchard_validation.geojson" in j][0], 'rb') as k:
            ref_Data = gpd.read_file(k, engine='pyogrio',use_arrow=True)
        
        return dem, nir, red, reg, rgb, points, mask, ref_Data

    # remove points that aren't in mask
    @staticmethod
    def _removePoints(geom, mask):
        # find points within mask
        return mask.contains(geom)

    @staticmethod
    def _recursivePointRemoval(geoms, mask):
        # remove points that touch the mask
        hold = []
        for i in range(0, len(geoms)):
            if collect._removePoints(geoms.iloc[i,1], mask)[0]:
                hold.append(i)
        return hold

    @staticmethod
    def _repprojectData(data, xds_match):
            data = data.rio.reproject_match(xds_match, resampling = Resampling.nearest) # Resampling.bilinear
            data = data.assign_coords({
                "x": xds_match.x,
                "y": xds_match.y,})
            return data

    # TODO: should be sped up from 20 seconds to less than 2 seconds
    # TODO: Something is wrong with blue and green bands, don't come out the same as others
    def fixData(self, tif, geojson, zipped):
        # Resamples data and removes irrelevant points
        dem, nir, red, reg, rgb, points, mask, ref_data = collect._retrieveData(tif, geojson, zipped)

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
        nir = collect._repprojectData(nir, xds_match)
        red = collect._repprojectData(red, xds_match)
        reg = collect._repprojectData(reg, xds_match)
        rgb = collect._repprojectData(rgb, xds_match)
        green = rgb[1]
        blue = rgb[2]

        data = {"dem": dem, 
                "nir": nir, 
                "red": red, 
                "reg": reg, 
                "rgb": rgb, 
                "green": green, 
                "blue": blue}
        

        index_mask_intersect = collect._recursivePointRemoval(points, mask)
        delineations = points.iloc[index_mask_intersect]
        delineations.reset_index(drop=True, inplace=True)

        return data, delineations, mask, ref_data

    # TODO: select better vegetative indices 
    def _vegIndices(self, data):
            # reg stands for red edge
        
        data["ndvi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
        data["ndre"] = (data["nir"] - data["reg"]) / (data["nir"] + data["reg"])
        data["gndvi"] = (data["nir"] - data["green"]) / (data["nir"] + data["green"])
        data["endvi"] = ((data["nir"]+ data["green"] - 2 * data["blue"]) / (data["nir"] + data["green"] + 2 * data["blue"]))
        data["intensity"] = data["nir"] + data["green"] + data["blue"]
        data["saturation"] = (data["intensity"] -3 * data["blue"]) / data["intensity"]
        #       as only NDVI seems to distinguish ground pixels from trees well
        #data["NDVI"].plot()
        #data["NDRE"].plot()
        #data["GNDVI"].plot()
        #data["ENDVI"].plot()

        # data["NDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/NDVI.tif")
        # data["NDRE"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/NDRE.tif")
        # data["GNDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/GNDVI.tif")
        # data["ENDVI"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/ENDVI.tif")
        # data["Intensity"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/Intensity.tif")
        # data["Satuartion"].rio.to_raster("C:/Users/balen/OneDrive/Desktop/Saturation.tif")
        self.spectralData = data
        #return data


# These are methods that called in the main script
# they are not needed to be in this clcass but work well here


# init method or constructor
# def __init__(self, sampleSize):
#     data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = self.collectFiles(sampleSize)
#     self.num = 0
#     self.data_paths_tif = data_paths_tif
#     self.data_paths_geojson = data_paths_geojson
#     self.data_paths_geojson_zipped = data_paths_geojson_zipped

# This init is just to handle unzipping geojsons
# We can make it in terms of just geojsons
def _dCollect(size, file_type):
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

def collectFiles(sampleSize = 20):
    # Collect file paths
    # for trial implementation
    # for final implementation, need to ask user to input file paths of 
    # interest
    data_paths_tif = _dCollect(size=sampleSize, file_type="tif")
    data_paths_geojson = _dCollect(size=sampleSize, file_type="geojson")
    data_paths_geojson_zipped = _dCollect(size=sampleSize, file_type="gz")
    random.seed(2024)
    # Create raster stack in 
    # this is a safe way to open zipped files without extracting them
    # import gzip
    # file = "C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/Data/93001/orchard_validation.geojson.gz"
    # with gzip.open(file, 'rb') as f:
    #     trys = gpd.read_file(f)
    return data_paths_tif, data_paths_geojson, data_paths_geojson_zipped