# %%
import joblib

data1 = joblib.load("results/training/data0_12.pkl")
data2 = joblib.load("results/training/data13_19.pkl")
data3 = joblib.load("results/training/data19_20.pkl")
data4 = joblib.load("results/training/data20_21.pkl")
data5 = joblib.load("results/training/data21_30.pkl")
data6 = joblib.load("results/training/data30_40.pkl")

data = data1['data'] + data2['data'] + data3['data'] + data4['data'] + data5['data'] + data6['data']

joblib.dump(data, "results/training/data0_40.pkl")
# %%
import joblib
data = joblib.load("results/training/data0_40.pkl")
data7 = joblib.load("results/training/data40_50.pkl")
data8 = joblib.load("results/training/data50_60.pkl")
data9 = joblib.load("results/training/data60_70.pkl")

data = data + data7['data'] + data8['data'] + data9['data']

joblib.dump(data, "results/training/datafull0_70.pkl")

# %%

import joblib
data = joblib.load("results/training/datafull0_70.pkl")

# TODO: Do this for entire dataset and save separately
# columns_to_remove = ["minor_axis", "radius_of_gyration", "major_axis"]
# for i in range(70):
#     data[i].drop(columns=columns_to_remove, inplace=True)

# joblib.dump(data, "results/training/data0_70.pkl")

# remove columns that are already justified
columns_to_remove = ["minor_axis", "radius_of_gyration", 'major_axis','circularity', 'eccentricity', 'GNDVI', 'NIR', 'Contrast', 'SAVI', 'z0']
for i in range(70):
    data[i].drop(columns=columns_to_remove, inplace=True)

joblib.dump(data, "results/training/datafull0_70.pkl")


# %%
# need to remove zernike

import joblib
data = joblib.load("results/training/data0_70.pkl")
columns_to_remove = ['z1', 'z2', 'z3', 'z4', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23', 'z24']
for i in range(70):
    data[i].drop(columns=columns_to_remove, inplace=True)

joblib.dump(data, "results/training/data0_70.pkl")

# %%

import joblib
import numpy as np
data = joblib.load("results/training/data0_70.pkl")


y = np.array(data[5].loc[:, "Y"]).T 
y = np.where(y == 'Outlier', 1, 0)

X = np.array(data[i].loc[:, "confidence":])
        
outliers_fraction = np.count_nonzero(y) / len(y) if np.count_nonzero(y) > 0 else 0.01


# %%

import joblib
data1 = joblib.load("results/testing/data70_80.pkl")
data2 = joblib.load("results/testing/data80_90.pkl")
data3 = joblib.load("results/testing/data90_97.pkl")
data4 = joblib.load("results/testing/data97_98.pkl")
data5 = joblib.load("results/testing/data98_101.pkl")

data = data1['data'] + data2['data'] + data3['data'] + data4['data'] + data5['data']
# %%

columns_to_remove = ['circularity', 'eccentricity', 'GNDVI', 'NIR', 'Contrast', 'SAVI', 'z0', 'z1', 'z2', 'z3', 'z4', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23', 'z24']
columns_to_remove = ['radius_of_gyration', 'minor_axis', 'major_axis']
for i in range(len(data)):
    data[i].drop(columns=columns_to_remove, inplace=True)

joblib.dump(data, "results/testing/data70_101.pkl")


# need to donwload masks and images
data1 = joblib.load("results/testing/data70_80.pkl")
data2 = joblib.load("results/testing/data80_90.pkl")
data3 = joblib.load("results/testing/data90_97.pkl")
data4 = joblib.load("results/testing/data97_98.pkl")
data5 = joblib.load("results/testing/data98_101.pkl")

masks = data1['mask'] + data2['mask'] + data3['mask'] + data4['mask'] + data5['mask']
images = [x['rgb'] for x in data1['spectralData']] + [x['rgb'] for x in data2['spectralData']] + [x['rgb'] for x in data3['spectralData']] + [x['rgb'] for x in data4['spectralData']] + [x['rgb'] for x in data5['spectralData']]

for i in range(len(images)):
    mask = masks[i]
    img = images[i][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    img = img/255

    images[i] = img

# joblib.dump(masks, "results/testing/masks70_101.pkl")
joblib.dump(images, "results/testing/images70_101.pkl")


# %%
import joblib
# need to donwload masks and images
data1 = joblib.load("results/training/data60_70.pkl")

masks = data1['mask']
images = [x['rgb'] for x in data1['spectralData']]

for i in range(len(images)):
    mask = masks[i]
    img = images[i][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    img = img/255

    images[i] = img

# joblib.dump(masks, "results/testing/masks70_101.pkl")
joblib.dump(images, "results/training/images60_70.pkl")
# %%
