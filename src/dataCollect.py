# %%
import os
import random
from glob import glob

def dCollect_Init(size):
    # For now unzip all jsons
    # From current directory get 
    os.chdir('Data/')
    pop_erf = os.listdir()

    # for now obtain a small subset of data from list of 316 files to test
    # Try 20 folders
    if size < 316:
        random.seed(2024)
        # obtain a random sample
        sample_erf = random.sample(pop_erf, size)
    else:
        sample_erf = pop_erf

    path_holder = []
    for files in sample_erf:
        path = []
        os.chdir(files + "\\")
        d = os.getcwd()
        # Read in files
        for x in os.listdir():
            if not x.endswith(".ini"):
                path.append(os.getcwd() +"\\"+  x )
            # Go back to previous directory
        os.chdir("..")
        path_holder.append(path)
    # Move back to src directory
    os.chdir("..")

    return path_holder

# %%

def dCollect(size):
    os.chdir('Data/')
    pop_erf = os.listdir()

    # for now obtain a small subset of data from list of 316 files to test
    # Try 20 folders
    if size < 316:
        random.seed(2024)
        # obtain a random sample
        sample_erf = random.sample(pop_erf, size)
    else:
        sample_erf = pop_erf

    path_holder_tif = []
    path_holder_geojson = []

    for files in sample_erf:
        path = []
        os.chdir(files + "\\")
        # Read in files
        path = glob(os.getcwd() +"/*.tif" )
        path_holder_tif.append(path)
        path = glob(os.getcwd() +"/*.geojson" )
        # Don't need a list of lists for this
        path_holder_geojson.append(path[0])    
        # Move back to Data Directory
        os.chdir("..")    

    # Move back to src directory
    os.chdir("..")




#a = dCollect(20)
# %%
