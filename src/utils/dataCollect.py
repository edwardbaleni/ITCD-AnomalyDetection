import os
import random
from glob import glob
import gzip
import geopandas as gpd
import xarray
import rioxarray as rio
from rasterio.enums import Resampling
import re

class collect:
    def __init__(self, num, tifs, geojsons, zips):
        self._num = num
        self._tif, self._geojson, self._zipped = tifs[self._num], geojsons[self._num], zips[self._num]
        self.spectralData, self.delineations, self.mask, self.ref_data = self.fixData(self._tif, self._geojson, self._zipped)      
        self._vegIndices(self.spectralData)  
        self.erf = (re.search("\d+", geojsons[num][0])).group(0)

    @staticmethod
    def _retrieveData(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
        """
        Retrieve and load various geospatial datasets from provided file paths.
        Parameters:
        data_paths_tif (list of str): List of file paths to .tif files containing different raster data.
        data_paths_geojson (list of str): List of file paths to .geojson files containing vector data.
        data_paths_geojson_zipped (list of str): List of file paths to zipped .geojson files containing vector data.
        Returns:
        tuple: A tuple containing the following elements:
            - dem (rasterio.io.DatasetReader): Digital Elevation Model raster data.
            - nir (rasterio.io.DatasetReader): Near-Infrared raster data.
            - red (rasterio.io.DatasetReader): Red band raster data.
            - reg (rasterio.io.DatasetReader): Red Edge raster data.
            - rgb (rasterio.io.DatasetReader): RGB raster data.
            - points (geopandas.GeoDataFrame): GeoDataFrame containing points data from mask_rcnn.geojson.
            - mask (geopandas.GeoDataFrame): GeoDataFrame containing survey polygon data.
            - ref_Data (geopandas.GeoDataFrame): GeoDataFrame containing orchard validation data.
        Raises:
        FileNotFoundError: If any of the specified files are not found.
        """
        
        nir = rio.open_rasterio([j for j in data_paths_tif if "nir_native" in j][0])
        red = rio.open_rasterio([j for j in data_paths_tif if "red_native" in j][0])
        reg = rio.open_rasterio([j for j in data_paths_tif if "reg_native" in j][0])
        rgb = rio.open_rasterio([j for j in data_paths_tif if "visible_5cm" in j][0])
        dem = rio.open_rasterio([j for j in data_paths_tif if "dem_native" in j][0])
        mask = gpd.read_file([j for j in data_paths_geojson if "survey_polygon" in j][0], engine='pyogrio',use_arrow=True)
        with gzip.open([j for j in data_paths_geojson_zipped if "mask_rcnn.geojson" in j][0], 'rb') as f:
            points = gpd.read_file(f, engine='pyogrio', use_arrow=True)
        with gzip.open([j for j in data_paths_geojson_zipped if "orchard_validation.geojson" in j][0], 'rb') as k:
            ref_Data = gpd.read_file(k, engine='pyogrio',use_arrow=True)
        
        return dem, nir, red, reg, rgb, points, mask, ref_Data

    @staticmethod
    def _removePoints(geom, mask):
        """
        Determine if the given geometry points are within the specified mask.
        Parameters:
        geom (shapely.geometry): The geometry object containing the points to be checked.
        mask (shapely.geometry): The geometry object representing the mask area.
        Returns:
        bool: True if the points in geom are within the mask, False otherwise.
        """

        # find points within mask
        return mask.contains(geom)

    @staticmethod
    def _recursivePointRemoval(geoms, mask):
        """
        Recursively removes points from geometries that touch a given mask.
        Args:
            geoms (GeoDataFrame): A GeoDataFrame containing geometries.
            mask (Geometry): A geometry used as a mask to determine which points to remove.
        Returns:
            list: A list of indices of the geometries that had points removed.
        """

        # remove points that touch the mask
        hold = []
        for i in range(0, len(geoms)):
            if collect._removePoints(geoms.loc[i,"geometry"], mask)[0]:
                hold.append(i)
        return hold

    @staticmethod
    def _repprojectData(data, xds_match):
        """
        Reprojects the given data to match the spatial resolution and coordinates of the provided xarray dataset.
        Parameters:
        data (xarray.DataArray or xarray.Dataset): The data to be reprojected.
        xds_match (xarray.DataArray or xarray.Dataset): The xarray object whose spatial resolution and coordinates will be matched.
        Returns:
        xarray.DataArray or xarray.Dataset: The reprojected data with updated coordinates.
        """    
        data = data.rio.reproject_match(xds_match, resampling = Resampling.nearest) # Resampling.bilinear
        data = data.assign_coords({
            "x": xds_match.x,
            "y": xds_match.y,})
        return data

    # TODO: should be sped up from 20 seconds to less than 2 seconds
    # TODO: Something is wrong with blue and green bands, don't come out the same as others
    def fixData(self, tif, geojson, zipped):
        """
        Processes and fixes the input data by resampling and removing irrelevant points.
        Args:
            tif (str): Path to the TIFF file containing raster data.
            geojson (str): Path to the GeoJSON file containing vector data.
            zipped (str): Path to the zipped file containing additional data.
        Returns:
            tuple: A tuple containing:
                - data (dict): A dictionary with the following keys:
                    - "dem": Digital Elevation Model (DEM) raster data.
                    - "nir": Near-Infrared (NIR) raster data.
                    - "red": Red band raster data.
                    - "reg": Red Edge band raster data.
                    - "rgb": RGB raster data.
                    - "green": Green band raster data extracted from RGB.
                    - "blue": Blue band raster data extracted from RGB.
                - delineations (DataFrame): A DataFrame containing the relevant points after mask intersection.
                - mask (array): The mask array used for point removal.
                - ref_data (any): Reference data retrieved from the input files.
        """

        # Resamples data and removes irrelevant points
        dem, nir, red, reg, rgb, points, mask, ref_data = collect._retrieveData(tif, geojson, zipped)

        xds_match = dem

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

    def _vegIndices(self, data):
        """
        Calculate various vegetation indices from the provided spectral data.
        This method computes several vegetation indices, which are useful for analyzing
        vegetation health and characteristics from spectral data. The indices calculated include:
        - NDVI (Normalized Difference Vegetation Index)
        - NDRE (Normalized Difference Red Edge)
        - GNDVI (Green Normalized Difference Vegetation Index)
        - ENDVI (Enhanced Normalized Difference Vegetation Index)
        - SAVI (Soil Adjusted Vegetation Index)
        - EVI (Enhanced Vegetation Index)
        - CI (Chlorophyll Index)
        - OSAVI (Optimized Soil Adjusted Vegetation Index)
        - SR_REG (Simple Ratio using Red Edge)
        Parameters:
        data (pandas.DataFrame): A DataFrame containing the spectral data with columns for 'nir', 'red', 'reg', 'green', and 'blue' bands.
        Returns:
        None: The method updates the input DataFrame in place by adding new columns for each vegetation index.
        """
        
        # reg stands for red edge
        
        data["ndvi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
        data["ndre"] = (data["nir"] - data["reg"]) / (data["nir"] + data["reg"])
        data["gndvi"] = (data["nir"] - data["green"]) / (data["nir"] + data["green"])
        data["endvi"] = ((data["nir"]+ data["green"] - 2 * data["blue"]) / (data["nir"] + data["green"] + 2 * data["blue"]))
        data["savi"] = ((data["nir"]/1000 - data["red"]/1000)/(data["nir"]/1000 + data["red"]/1000 + 0.5))*(1+0.5)
        data["evi"] = 2.5 * ((data["nir"]/1000 - data["red"]/1000)/(data["nir"]/1000 + 6*data["red"]/1000 -7.5 * data["blue"]/1000 +1))
        data["ci"] = (data["nir"]/1000) / (data["reg"]/1000) - 1 
        data["osavi"] = 1.16 * (data["nir"]/1000 - data["red"]/1000) / (data["nir"]/1000+data["red"]/1000 + 0.16)
        data["sr_reg"] = data["nir"]/data["reg"]
        # Graveyard of unworkeable vegetative indices.
        # When plotted, they did not distinguish well between soil and 
        # vegetation
        # data["intensity"] = data["nir"] + data["green"] + data["blue"]
        # data["saturation"] = (data["intensity"] -3 * data["blue"]) / data["intensity"]
        # msavi = (2 * spectralData["nir"]/1000 + 1 - np.sqrt( ( 2 * spectralData["nir"]/1000 + 1)**2 - 8 * (spectralData["nir"]/1000 - spectralData["red"]/1000))) / 2
        # data["ccci"] = data["ndre"]/data["ndvi"]
        # sr = spectralData["nir"]/spectralData["red"] # Simple Ratio
        # WDRI = sr = 0.1* spectralData["nir"] - spectralData["red"] / (0.1 * spectralData["nir"]) + spectralData["red"] # Wide Dynamic Range Vegetation Index
        # vari = (spectralData["green"] - spectralData["red"])/(spectralData["green"] + spectralData["red"] -spectralData["blue"]) 
        # ci = (spectralData["nir"]/1000) / (spectralData["green"]/1000) - 1


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

def _dCollect(size, file_type):
    """
    Collects a specified number of file paths of a given type from a directory.
    This function changes the current working directory to 'Data/', lists all files
    in the directory, and then selects a random sample of files if the specified size
    is less than the total number of files. It then collects the paths of files of the
    specified type from the sampled directories.
    Args:
        size (int): The number of directories to sample. If size is greater than or equal
                    to the total number of directories, all directories are used.
        file_type (str): The file extension/type to look for within the sampled directories.
    Returns:
        list: A list of lists, where each inner list contains the paths of files of the
              specified type from a sampled directory.
    """

    # From current directory get 
    os.chdir('Data/')
    pop_erf = os.listdir()

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

def collectFiles(sampleSize = 20):
    """
    Collects file paths for different file types and returns them.
    This function collects file paths for TIFF, GeoJSON, and zipped GeoJSON files.
    It uses a helper function `_dCollect` to gather the file paths. The function
    also sets a random seed for reproducibility.
    Parameters:
    sampleSize (int): The number of file paths to collect for each file type. Default is 20.
    Returns:
    tuple: A tuple containing three lists:
        - data_paths_tif (list): List of file paths for TIFF files.
        - data_paths_geojson (list): List of file paths for GeoJSON files.
        - data_paths_geojson_zipped (list): List of file paths for zipped GeoJSON files.
    """

    data_paths_tif = _dCollect(size=sampleSize, file_type="tif")
    data_paths_geojson = _dCollect(size=sampleSize, file_type="geojson")
    data_paths_geojson_zipped = _dCollect(size=sampleSize, file_type="gz")
    random.seed(2024)

    return data_paths_tif, data_paths_geojson, data_paths_geojson_zipped