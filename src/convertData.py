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
data = joblib.load("results/training/data0_40.pkl")
data7 = joblib.load("results/training/data40_50.pkl")
data8 = joblib.load("results/training/data50_60.pkl")
data9 = joblib.load("results/training/data60_70.pkl")

data = data + data7['data'] + data8['data'] + data9['data']

joblib.dump(data, "results/training/data0_70.pkl")

# %%

import joblib
data = joblib.load("results/training/data0_70.pkl")

# TODO: Do this for entire dataset and save separately
# columns_to_remove = ["minor_axis", "radius_of_gyration", "major_axis"]
# for i in range(70):
#     data[i].drop(columns=columns_to_remove, inplace=True)

# joblib.dump(data, "results/training/data0_70.pkl")

# remove columns that are already justified
columns_to_remove = ['circularity', 'eccentricity', 'GNDVI', 'NIR', 'Contrast', 'SAVI', 'z0']
for i in range(70):
    data[i].drop(columns=columns_to_remove, inplace=True)

joblib.dump(data, "results/training/data0_70.pkl")


# %%
# need to remove zernike

import joblib
data = joblib.load("results/training/data0_70.pkl")
columns_to_remove = ['z1', 'z2', 'z3', 'z4', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23', 'z24']
for i in range(70):
    data[i].drop(columns=columns_to_remove, inplace=True)

joblib.dump(data, "results/training/data0_70.pkl")