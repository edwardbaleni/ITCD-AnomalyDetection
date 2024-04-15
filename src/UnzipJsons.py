# %%

import gzip
import shutil
import dataCollect

# For now unzip all jsons
# https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
def read_in(file):
    with gzip.open(file, "rb") as f_in:
        with open(file[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in,f_out)

if __name__ == "__main__":
    data_list = dataCollect.dCollect(size = 316)

    for i in range(len(data_list)):
         read_in(data_list[i])