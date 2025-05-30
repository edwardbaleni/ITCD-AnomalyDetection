# %%
import joblib
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from Model import Geary

from utils.plotAnomaly import plotScores, plot


from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns

# %%

data = joblib.load("results/testing/data70_101.pkl")
images = joblib.load("results/testing/images70_101.pkl")

# %%

#     img = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
#     img = img/255

# need to donwload masks and images

# lof = pd.read_csv("results/meta/Test/LOF_OOS_params.csv")
# abod = pd.read_csv("results/meta/Test/ABOD_OOS_params.csv")
# iforest = pd.read_csv("results/meta/Test/IF_OOS_params.csv")
# pca = pd.read_csv("results/meta/Test/PCA_OOS_params.csv")
# # Geary
# # ECOD

# # Melt the dataframe and drop NA values
# melted_lof = lof.melt(id_vars=['Orchard', 'Predicted AP', 'params_n_neighbors'],
#                       value_vars=['chebyshev', 'euclidean', 'manhattan', 'minkowski']).dropna()
# # Filter out rows where the value is 0 and reset the index
# melted_lof.rename(columns={'variable': 'metric'}, inplace=True)
# lof = melted_lof[melted_lof['value'] != 0].reset_index(drop=True)
# del lof['value']

# ap_scores = pd.DataFrame()
# auc_scores = pd.DataFrame()


# # perform outlier detection
# for i in range(len(data)): 
#     print(f"Orchard {i + 71}")
#     print("========================================")
#     scaler = RobustScaler()
#     dataset = scaler.fit_transform(data[i].loc[:,'confidence':])
#     y = np.array(data[i].loc[:, "Y"]).T 
#     y = np.where(y == 'Outlier', 1, 0)
#     # LOF
#     clf = LOF(contamination=0.05, n_neighbors=int(lof['params_n_neighbors'][i]), metric=lof['metric'][i])
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_

#     ap_lof = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         auc_lof = 0
#     else:
#         auc_lof = roc_auc_score(y, y_test_scores)
#     y_test_scores = np.interp(y_test_scores, (y_test_scores.min(), y_test_scores.max()), (0, 1))
#     plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/lof_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/lof_{i+71}.png")
#     # print(f'LOF - AP: {ap_lof}, Predicted AP: {lof["Predicted AP"][i]}, AUC: {auc_lof}')


#     # # ABOD
#     clf = ABOD(contamination=0.05, n_neighbors=int(abod['params_n_neighbors'][i]))
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     abod_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         abod_auc = 0
#     else:
#         abod_auc = roc_auc_score(y, y_test_scores)
#     y_test_scores = np.interp(y_test_scores, (y_test_scores.min(), y_test_scores.max()), (0, 1))

#     plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/abod_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/abod_{i+71}.png")
#     # print(f'ABOD - AP: {abod_ap}, Predicted AP: {abod["Predicted AP"][i]}, AUC: {abod_auc}')


#     # Isolation Forest
#     clf = IForest(contamination=0.05,n_estimators=int(iforest['params_n_estimators'][i]), max_features=int(iforest['params_max_features'][i]))
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     iforest_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         iforest_auc = 0
#     else:
#         iforest_auc = roc_auc_score(y, y_test_scores)
    
#     # TODO: Convert IForest scores from [0, 0.25] to [0, 1]
#     y_test_scores = np.interp(y_test_scores, (y_test_scores.min(), y_test_scores.max()), (0, 1))
#     # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/iforest_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/iforest_{i+71}.png")
#     # print(f'Iforest - AP: {iforest_ap}, Predicted AP: {abod["Predicted AP"][i]}, AUC: {abod_auc}')

#     # # PCA
#     clf = PCA(contamination=0.05, n_selected_components=int(pca['params_n_selected_components'][i]))
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     pca_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         pca_auc = 0
#     else:
#         pca_auc = roc_auc_score(y, y_test_scores)
#     y_test_scores = np.interp(y_test_scores, (y_test_scores.min(), y_test_scores.max()), (0, 1))
#     plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/pca_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/pca_{i+71}.png")
#     # print(f'PCA - AP: {average_precision_score(y, y_test_scores)}, Predicted AP: {pca["Predicted AP"][i]}, AUC: {pca_auc}')


#     # # ECOD
#     clf = ECOD(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     ecod_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         ecod_auc = 0
#     else:
#         ecod_auc = roc_auc_score(y, y_test_scores)
#     y_test_scores = np.interp(y_test_scores, (y_test_scores.min(), y_test_scores.max()), (0, 1))
#     plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/ecod_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/ecod_{i+71}.png")
#     # print(f'ECOD - AP: {ecod_ap}', f'AUC: {ecod_auc}')


#     # # Geary
#     clf = Geary(contamination=0.05,
#                 geometry=data[i]["geometry"], 
#                 centroid=data[i]["centroid"])
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     geary_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         geary_auc = 0
#     else:
#         geary_auc = roc_auc_score(y, y_test_scores)
#     # plotScores(images[i], data[i], y_test_scores, f"results/oos/out_scores/geary_{i+71}.png")
#     # plot(images[i], data[i][y_labels == 0], data[i][y_labels == 1], f"results/oos/outliers/geary_{i+71}.png")
#     # print(f'Geary - AP: {geary_ap}, AUC: {geary_auc}')    
    
#     # # obtain AUC and AP
#     auc_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": auc_lof, "ABOD": abod_auc, "IF": iforest_auc, "PCA": pca_auc, "ECOD": ecod_auc, "GBOD": geary_auc}, orient='index').T
#     auc_scores = pd.concat([auc_scores, auc_data])
#     ap_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": ap_lof, "ABOD": abod_ap, "IF": iforest_ap, "PCA": pca_ap, "ECOD": ecod_ap, "GBOD": geary_ap}, orient='index').T
#     ap_scores = pd.concat([ap_scores, ap_data])




# # %%
# ap_scores.reset_index(drop=True, inplace=True)
# auc_scores.reset_index(drop=True, inplace=True)
# ap_scores.to_csv("results/oos/ap_scores.csv")
# auc_scores.to_csv("results/oos/auc_scores.csv")


# %%
ap_scores = pd.read_csv("results/oos/ap_scores.csv")
auc_scores = pd.read_csv("results/oos/auc_scores.csv")

# Calculate column averages for AP scores and AUC scores
ap_avg = ap_scores.loc[:,"LOF":].mean(axis=0)
auc_avg = auc_scores.loc[:, "LOF":].mean(axis=0)

# Print the averages
print("Average AP scores:")
print(ap_avg)
print("\nAverage AUC scores:")
print(auc_avg)

sum_ap = ap_scores.loc[:,"LOF":].std(axis=0)
sum_auc = auc_scores.loc[:, "LOF":].std(axis=0)
print("Default std AP scores:")
print(sum_ap)
print("\n Default std AUC scores:")
print(sum_auc)







# # %%

# # TODO: Demonstrate the above on default settings


# ap_scores = pd.DataFrame()
# auc_scores = pd.DataFrame()


# # perform outlier detection
# for i in range(len(data)):
#     print(f"Orchard {i + 71}")
#     print("========================================")
#     scaler = RobustScaler()
#     dataset = scaler.fit_transform(data[i].loc[:,'confidence':])
#     y = np.array(data[i].loc[:, "Y"]).T 
#     y = np.where(y == 'Outlier', 1, 0)
#     # LOF
#     clf = LOF(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_

#     ap_lof = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         auc_lof = 0
#     else:
#         auc_lof = roc_auc_score(y, y_test_scores)

#     # # ABOD
#     clf = ABOD(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     abod_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         abod_auc = 0
#     else:
#         abod_auc = roc_auc_score(y, y_test_scores)

#     # Isolation Forest
#     clf = IForest(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     iforest_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         iforest_auc = 0
#     else:
#         iforest_auc = roc_auc_score(y, y_test_scores)
    
#     # # PCA
#     clf = PCA(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     pca_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         pca_auc = 0
#     else:
#         pca_auc = roc_auc_score(y, y_test_scores)

#     # # ECOD
#     clf = ECOD(contamination=0.05)
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     ecod_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         ecod_auc = 0
#     else:
#         ecod_auc = roc_auc_score(y, y_test_scores)


#     # # Geary
#     clf = Geary(contamination=0.05,
#                 geometry=data[i]["geometry"], 
#                 centroid=data[i]["centroid"])
#     clf.fit(dataset)
#     y_test_scores = clf.decision_scores_
#     y_labels = clf.labels_
#     geary_ap = average_precision_score(y, y_test_scores)
#     if np.sum(y) == 0:
#         geary_auc = 0
#     else:
#         geary_auc = roc_auc_score(y, y_test_scores)
    
#     # # obtain AUC and AP
#     auc_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": auc_lof, "ABOD": abod_auc, "IF": iforest_auc, "PCA": pca_auc, "ECOD": ecod_auc, "GBOD": geary_auc}, orient='index').T
#     auc_scores = pd.concat([auc_scores, auc_data])
#     ap_data = pd.DataFrame.from_dict({"Orchard":f'Orchard {i + 71}' ,"LOF": ap_lof, "ABOD": abod_ap, "IF": iforest_ap, "PCA": pca_ap, "ECOD": ecod_ap, "GBOD": geary_ap}, orient='index').T
#     ap_scores = pd.concat([ap_scores, ap_data])


# # %%
# ap_scores.reset_index(drop=True, inplace=True)
# auc_scores.reset_index(drop=True, inplace=True)
# ap_scores.to_csv("results/oos/default_ap_scores.csv")
# auc_scores.to_csv("results/oos/default_auc_scores.csv")


# %%
def_ap_scores = pd.read_csv("results/oos/default_ap_scores.csv")
def_auc_scores = pd.read_csv("results/oos/default_auc_scores.csv")

# Calculate column averages for AP scores and AUC scores
def_ap_avg = def_ap_scores.loc[:,"LOF":].mean(axis=0)
def_auc_avg = def_auc_scores.loc[:, "LOF":].mean(axis=0)

# Print the averages
print("Default Average AP scores:")
print(def_ap_avg)
print("\n Default Average AUC scores:")
print(def_auc_avg)

# %%

import matplotlib.pyplot as plt

ap_scores.drop(ap_scores.columns[0], axis=1, inplace=True)
def_ap_scores.drop(def_ap_scores.columns[0], axis=1, inplace=True)

ap_scores.reset_index(drop=True, inplace=True)
def_ap_scores.reset_index(drop=True, inplace=True)

# %%

# Rename 'IF' to 'IForest' in the AP and AUC scores dataframes
ap_scores.rename(columns={'IF': 'IForest'}, inplace=True)
def_ap_scores.rename(columns={'IF': 'IForest'}, inplace=True)
auc_scores.rename(columns={'IF': 'IForest'}, inplace=True)
def_auc_scores.rename(columns={'IF': 'IForest'}, inplace=True)

ap_scores.rename(columns={'Geary': 'GBOD'}, inplace=True)
def_ap_scores.rename(columns={'Geary': 'GBOD'}, inplace=True)
auc_scores.rename(columns={'Geary': 'GBOD'}, inplace=True)
def_auc_scores.rename(columns={'Geary': 'GBOD'}, inplace=True)

# %%
# Melt the dataframes for easier plotting
ap_scores_melted = ap_scores.melt(id_vars=['Orchard'], var_name='Estimator', value_name='AP Score')
def_ap_scores_melted = def_ap_scores.melt(id_vars=['Orchard'], var_name='Estimator', value_name='AP Score')


# Add a column to distinguish between the datasets
ap_scores_melted['Dataset'] = 'HPOD'
def_ap_scores_melted['Dataset'] = 'Default'

# Concatenate the dataframes
combined_ap_scores = pd.concat([ap_scores_melted, def_ap_scores_melted])
# Plot the boxplots
plt.figure(figsize=(20, 12))
sns.boxplot(x='Estimator', y='AP Score', hue='Dataset', data=combined_ap_scores)
# plt.title('AP Scores by Estimator and Dataset')
plt.legend(title='Hyperparameter State', fontsize=20, title_fontsize=25, loc='upper right')
plt.xlabel('Estimator', fontsize=25)
plt.ylabel('AP Score', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("results/oos/ap_scores_boxplot.png")
plt.show()
# %%

# Melt the dataframes for easier plotting
auc_scores.drop(auc_scores.columns[0], axis=1, inplace=True)
def_auc_scores.drop(def_auc_scores.columns[0], axis=1, inplace=True)

auc_scores.reset_index(drop=True, inplace=True)
def_auc_scores.reset_index(drop=True, inplace=True)

# %%
# Melt the dataframes for easier plotting
auc_scores_melted = auc_scores.melt(id_vars=['Orchard'], var_name='Estimator', value_name='AUC Score')
def_auc_scores_melted = def_auc_scores.melt(id_vars=['Orchard'], var_name='Estimator', value_name='AUC Score')


# Add a column to distinguish between the datasets
auc_scores_melted['Dataset'] = 'HPOD'
def_auc_scores_melted['Dataset'] = 'Default'

# Concatenate the dataframes
combined_auc_scores = pd.concat([auc_scores_melted, def_auc_scores_melted])
# Plot the boxplots
plt.figure(figsize=(20, 12))
sns.boxplot(x='Estimator', y='AUC Score', hue='Dataset', data=combined_auc_scores)
# plt.title('AUC Scores by Estimator and Dataset')
plt.legend(title='Hyperparameter State', fontsize=20, title_fontsize=25, loc='lower right')
plt.xlabel('Estimator', fontsize=25)
plt.ylabel('AUC Score', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("results/oos/auc_scores_boxplot.png")
plt.show()
# %%


# Calculate column averages for AP scores and AUC scores
def_ap_avg = def_ap_scores.loc[:,"LOF":].mean(axis=0)
def_auc_avg = def_auc_scores.loc[:, "LOF":].mean(axis=0)

# Print the averages
print("Default Average AP scores:")
print(def_ap_avg)
print("\n Default Average AUC scores:")
print(def_auc_avg)

res_ap = pd.read_csv("C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/results/transductive/ap.csv")
res_auc = pd.read_csv("C:/Users/balen/OneDrive/Desktop/Git/Dissertation-AnomalyDetection/Dissertation-AnomalyDetection/src/results/transductive/auc.csv")
sum_ap = res_ap.loc[:,"ABOD":].std(axis=0)
sum_auc = res_auc.loc[:, "ABOD":].std(axis=0)
print("Default std AP scores:")
print(sum_ap)
print("\n Default std AUC scores:")
print(sum_auc)

# %%