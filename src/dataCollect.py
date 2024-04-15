# %%
import os
import random
from glob import glob

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
        random.seed(2024)
        # obtain a random sample
        sample_erf = random.sample(pop_erf, size)
    else:
        sample_erf = pop_erf

    path_holder = []
    for files in sample_erf:
        path = []
        os.chdir(files + "\\")
        # Read in files
        path = glob(os.getcwd() + "/*." + file_type )
        if file_type == "tif":
            path_holder.append(path)
        else:
            # Don't need a list of lists for this
            path_holder.append(path[0])    
        # Move back to Data Directory
        os.chdir("..")    
    
    # Move back to src directory
    os.chdir("..")

    return path_holder

# %%
# unzip folders
a = dCollect(size = 20, file_type = "gz")
