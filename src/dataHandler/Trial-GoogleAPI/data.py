# %%
import os
import shutil
import zipfile
import geopandas as gpd
import rasterio
from rasterio.plot import show

data = gpd.read_file("mask_rcnn.geojson")
img = rasterio.open("red_native.tif")

# %%
show(img)

data.head()
data.plot()


# %%

os.chdir('data/')
a = os.listdir()

# %%
path = []
for files in a:
    os.chdir(files + "\\UCT_Data_Project2")
    d = os.getcwd()
    # Read in files
    for x in os.listdir():
        path.append(os.getcwd() +"\\"+  x + "\\")
    # Go back to previous directory
    os.chdir("..")
    os.chdir("..")
# Move back to src directory
os.chdir("..")
print(path)


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

# %%
             
from pydrive2.auth import GoogleAuth
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
#gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", SCOPES) # assign credential file to gauth credentials attribute
#gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)# now make instance of google drive by passing gauth instance after assigning credentials now you are good to go

# %%
