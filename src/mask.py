# %%
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def getMaskBounds(comp, shape):
    # https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry
    # enclose this in a function

        # Obtain example file
    # a = gpd.GeoDataFrame.from_file(data_paths_geojson[1])
    # point = list(a.iloc[:,1])
    point = list(shape.iloc[:,1])

        # Combine all intersecting polygons
    point = shapely.unary_union(point)
        # Combine all non-intersecting polygons
    boundary = shapely.coverage_union_all(point)
        # Get the convex hull of the entire area (different from concave hull)
        # concave hull will get all the indents
        # will get more specific boundary with concave
        # maybe go with concave
        # Concave actually doesn't work well at all
        # However, convex is not perfect.
        # Concave works when you change the ratio
    boundary_periph = shapely.concave_hull(boundary, ratio=0.15)#shapely.convex_hull(boundary)
    #poly = gpd.GeoSeries(boundary_periph, crs=RGBs[0].crs.data)
    poly = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary_periph), crs=comp.crs.data)

    return (poly)

    # Masks data and gives all data the same transform
    # https://gis.stackexchange.com/questions/254766/update-geotiff-metadata-with-rasterio
    # The website above shows how one can update the metadata of a raster file
def getMask(rast, shape, placeHolder, out_tif="C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out.tif"):
    # Obtain masking, in terms of NIR file
    poly = getMaskBounds(comp=rast, shape=shape)
    # Get features for masking
    coords = getFeatures(poly)
    # Get information to output masking
    out_img, out_transform = mask(rast, shapes=coords, crop=True, all_touched=True, pad = True)
    
    # The meta data has to be the same as 
    out_meta = placeHolder.meta.copy()
    # Change transform here
        #
        #  The idea is to change the transform to the same as the bands
        #  But this means that the geojson file will be misaligned 
        #  For now keep comp=rast on line 124 instead of placeHolder 
        #
    #out_transform = placeHolder.transform
    print(out_meta)
    #epsg_code = int(rast.crs.data['init'][5:])
    #print(epsg_code)

    out_meta.update({"driver": "GTiff",
                    "count": rast.count,
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform })#,
     #               "crs": epsg_code})

    #out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out2.tif"
    with rio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)



# in a try - except, try mask , except create mask, and apply mask. 
# If we can't find masked file in data2 then mask it and obtain else obtain


# At this stage the masking is fine, 
# The transform is equal over all bands, however, the width and height isn't the same for each
# Everything except the width and height is good. Now we need to fix this
# Output all results to data2

# %%

#getMask(NIRs[0], Points[0], NIRs[0],out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out1.tif")
#getMask(DEMs[0], Points[0], NIRs[0],out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out2.tif")
getMask(RGBs[ num ], Points[ num ], RGBs[ num ],out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out.tif")

# %%
out_tif = "C:\\Users\\balen\\OneDrive\\Desktop\\Git\\Dissertation-AnomalyDetection\\Dissertation-AnomalyDetection\\src\\out.tif"
clipped = rio.open(out_tif)
show((clipped), cmap='terrain')
#es._stack_bands([clipped, clipped1])

# %%
# Plot them
fig, ax = plt.subplots(figsize=(15, 15))
rio.plot.show(clipped, ax=ax)
Points[ num ].plot(ax=ax, facecolor='none', edgecolor='blue')

# fig, ax = plt.subplots(figsize=(15, 15))
# rio.plot.show(RGBs[0], ax=ax)