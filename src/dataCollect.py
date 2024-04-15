import os
import random

def dCollect(size):
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

