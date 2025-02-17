import utils
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import utils.plotAnomaly as plot

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    sampleSize = 70
    data = []
    delineations = []
    mask = []
    spectralData = []
    erf_num = []
    refData = []

    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)

    # Obtain number of outliers each orchard and number of delineations
    OutlierInfo = pd.DataFrame(columns=["Orchard", "Outliers", "Delineations", "Ratio"])

    mn = 21
    mx = 30

    # TODO: Collect Test Set
    for num in range(mn, mx):
        myData = utils.engineer(num, 
                                data_paths_tif, 
                                data_paths_geojson, 
                                data_paths_geojson_zipped,
                                False)

        # obtain data for each of the 30 datasets
        data.append(myData.data.copy(deep=True))
        delineations.append(myData.delineations.copy(deep=True))
        mask.append(myData.mask.copy(deep=True))
        spectralData.append(myData.spectralData)
        erf_num.append(myData.erf)
        refData.append(myData.ref_data.copy(deep=True))

        # For plotting
        img = myData.spectralData["rgb"][0:3].rio.clip(myData.mask.geometry.values, myData.mask.crs, drop=True, invert=False)
        img = img/255

        # Plot all the orchards
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.axis('off')
        img.plot.imshow(ax=ax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.title("")
        fig.savefig("results/EDA/Orchards/orchard_{}.png".format(num+1))
        plot.plotRef(img, myData.data, "results/EDA/Orchards/reference_{}.png".format(num+1))

        new_row = {"Orchard": "Orchard {}".format(num+1), 
                    "Outliers": myData.data.loc[myData.data["Y"] == "Outlier"].shape[0], 
                    "Delineations": myData.data.shape[0]}
        new_row["Ratio"] = new_row["Outliers"]/new_row["Delineations"]
        OutlierInfo = pd.concat([OutlierInfo, pd.DataFrame([new_row])], ignore_index=True)

        print(f"Orchard {num+1} done")

    # OutlierInfo.to_csv("results/EDA/benchmark_data0_70.csv", index=False)

    with open(f'results/training/data{mn}_{mx}.pkl', 'wb') as f:
        joblib.dump({
            'data': data,
            'delineations': delineations,
            'mask': mask,
            'spectralData': spectralData,
            'erf_num': erf_num,
            'refData': refData
        }, f)

