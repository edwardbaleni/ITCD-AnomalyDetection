# %%
import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA
import utils.Triangulation as tri
import esda
from Model import Geary
from Model import transductive
from multiprocessing import cpu_count, Pool
import joblib
from os import listdir

from utils import cdDiagram
import seaborn as sns

# def getData():
#     data = joblib.load("results/training/data0_70.pkl")
#     path = "results/hyperparameter/"
#     home = listdir(path)

#     abod_files = [f'{path}{file}' for file in home if "ABOD" in file]
#     lof_files = [f'{path}{file}' for file in home if "LOF" in file]
#     pca_files = [f'{path}{file}' for file in home if "PCA" in file]
#     if_files = [f'{path}{file}' for file in home if "IF" in file]

#     return data, [joblib.load(file) for file in lof_files], [joblib.load(file) for file in abod_files], [joblib.load(file) for file in pca_files], [joblib.load(file) for file in if_files]


# def process(data, erf_num, lof_params, abod_params, pca_params, if_params):

#     return transductive.transductionResults(data, 
#                                                str(erf_num), 
#                                                lof_params, 
#                                                abod_params, 
#                                                pca_params, 
#                                                if_params)
# # %%
# data, LOF, ABOD, PCA, IF = getData()
# sampleSize = len(LOF)

# aucroc = pd.DataFrame()
# ap = pd.DataFrame()
# time = pd.DataFrame()

# for i in range(sampleSize):
#     y = np.array(data[i].loc[:, "Y"]).T 
#     y = np.where(y == 'Outlier', 1, 0)
#     if np.count_nonzero(y) == 0:
#         # continue
#         del data[i]

#     auc, average_precision, tme = process(data[i], 
#                       i, 
#                       LOF[i].best_params, 
#                       ABOD[i].best_params, 
#                       PCA[i].best_params, 
#                       IF[i].best_params)
    
#     aucroc = pd.concat([aucroc, auc], axis=0)
#     ap = pd.concat([ap, average_precision], axis=0)
#     time = pd.concat([time, tme], axis=0)
    
# aucroc.reset_index(drop=True, inplace=True)
# ap.reset_index(drop=True, inplace=True)
# time.reset_index(drop=True, inplace=True)

# %%
# Save aucroc with only unique indices
aucroc = pd.read_csv("results/transductive/auc.csv")
ap = pd.read_csv("results/transductive/ap.csv")
time = pd.read_csv("results/transductive/time.csv")

# standard deviations can be found in pickle files
# time.to_csv('results/transductive/time.csv', index=False, float_format='%.3f')
# aucroc.to_csv('results/transductive/auc.csv', index=False, float_format='%.3f')
# ap.to_csv('results/transductive/ap.csv', index=False, float_format='%.3f')

# These dataframes need to be in long format for the CD diagram
aucroc_long = pd.melt(aucroc, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
ap_long = pd.melt(ap, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
time_long = pd.melt(time,
                    id_vars=["Data", "# Samples"], 
                    value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

# Plot the CD diagram

aucroc_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
ap_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
time_long.columns = ['dataset_name', 'Observations', 'classifier_name', 'time']

aucroc_long = aucroc_long[['classifier_name', 'dataset_name', 'accuracy']]
ap_long = ap_long[['classifier_name', 'dataset_name', 'accuracy']]
time_long = time_long[['classifier_name','Observations', 'dataset_name', 'time']]


cdDiagram.draw_cd_diagram(df_perf=ap_long, title='', labels=True, measure = '/transductive/AP/AP')

cdDiagram.draw_cd_diagram(df_perf=aucroc_long, title='', labels=True, measure = '/transductive/AUCROC/AUC')

# cdDiagram.draw_cd_diagram(df_perf=time_long, title='', labels=True, measure = '/transductive/Time/Time')
# %%

aucroc.loc[:, 'ABOD':].mean()
ap.loc[:, 'ABOD':].mean()

# %%
        # pink       brown      yellow      lime       teal     blue
colors = ["#ec1763", "#A25C43", "#FABE37", "#91c059", "#118D92", "#204ecf"]
# Alternative with more detailed visualization
plt.figure(figsize=(20, 12))
sns.boxplot(x='classifier_name', y='accuracy', data=aucroc_long, palette=colors)
sns.stripplot(x='classifier_name', y='accuracy', data=aucroc_long, 
              size=10, color='.3', linewidth=0, alpha=0.6)

import matplotlib.patches as mpatches
classifiers = aucroc_long['classifier_name'].unique()
patches = [mpatches.Patch(color=colors[i], label=classifier) for i, classifier in enumerate(classifiers)]
plt.legend(handles=patches, title='Classifier', loc='best', prop={'size': 25}, title_fontsize=20)

plt.ylabel('AUCROC', fontsize=30)
plt.xlabel('', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
plt.savefig('results/transductive/AUCROC/AUC_Summary.png')
plt.show()

# %%
plt.figure(figsize=(20, 12))
sns.boxplot(x='classifier_name', y='accuracy', data=ap_long, palette=colors)
sns.stripplot(x='classifier_name', y='accuracy', data=ap_long, 
              size=10, color='.3', linewidth=0, alpha=0.6)

import matplotlib.patches as mpatches
classifiers = ap_long['classifier_name'].unique()
patches = [mpatches.Patch(color=colors[i], label=classifier) for i, classifier in enumerate(classifiers)]
plt.legend(handles=patches, title='Classifier', loc='best', prop={'size': 25}, title_fontsize=20)

plt.ylabel('Average Precision', fontsize=30)
plt.xlabel('', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
plt.savefig('results/transductive/AP/AP_Summary.png')
plt.show()


# %%


# XXX: Get CD diagrams for orchards with majority local outliers.

local = [0, 3, 5,6,15, 17, 20, 28, 30, 31, 33, 35, 44, 63, 64, 65]
ap_local = ap.iloc[local]
aucroc_local = aucroc.iloc[local]

ap_local_long = pd.melt(ap_local, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

aucroc_local_long = pd.melt(aucroc_local,
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

ap_local_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
aucroc_local_long.columns = ['dataset_name', 'classifier_name', 'accuracy']

ap_local_long = ap_local_long[['classifier_name', 'dataset_name', 'accuracy']]
aucroc_local_long = aucroc_local_long[['classifier_name', 'dataset_name', 'accuracy']]


cdDiagram.draw_cd_diagram(df_perf=ap_local_long, title='', labels=True, measure = '/transductive/AP/AP-local',num=16)
cdDiagram.draw_cd_diagram(df_perf=aucroc_local_long, title='', labels=True, measure = '/transductive/AUCROC/AUC-local',num=16)


# XXX: Get CD diagrams for orchards with majority global outliers.

# %%
glob = [1,2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 23,24, 25, 26, 27, 29, 32, 36, 37, 40, 41, 42, 43,45, 47, 48, 50,51,52,53,54,55, 56, 57, 58, 60, 61, 62, 66]

ap_global = ap.iloc[glob]
aucroc_global = aucroc.iloc[glob]

ap_global_long = pd.melt(ap_global, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
aucroc_global_long = pd.melt(aucroc_global,
                            id_vars=["Data"], 
                            value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

ap_global_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
aucroc_global_long.columns = ['dataset_name', 'classifier_name', 'accuracy']

ap_global_long = ap_global_long[['classifier_name', 'dataset_name', 'accuracy']]
aucroc_global_long = aucroc_global_long[['classifier_name', 'dataset_name', 'accuracy']]

cdDiagram.draw_cd_diagram(df_perf=ap_global_long, title='', labels=True, measure = '/transductive/AP/AP-global')
cdDiagram.draw_cd_diagram(df_perf=aucroc_global_long, title='', labels=True, measure = '/transductive/AUCROC/AUC-global')

# %%
# XXX: Get CD diagrams for orchards with mixed outliers.

mix = [i for i in range(67) if i not in local and i not in glob]

ap_mix = ap.iloc[mix]
aucroc_mix = aucroc.iloc[mix]

ap_mix_long = pd.melt(ap_mix, 
                      id_vars=["Data"], 
                      value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
aucroc_mix_long = pd.melt(aucroc_mix,
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

ap_mix_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
aucroc_mix_long.columns = ['dataset_name', 'classifier_name', 'accuracy']

ap_mix_long = ap_mix_long[['classifier_name', 'dataset_name', 'accuracy']]
aucroc_mix_long = aucroc_mix_long[['classifier_name', 'dataset_name', 'accuracy']]

try:
    cdDiagram.draw_cd_diagram(df_perf=ap_mix_long, title='', labels=True, measure = '/transductive/AP/AP-mix', num=14)
    cdDiagram.draw_cd_diagram(df_perf=aucroc_mix_long, title='', labels=True, measure = '/transductive/AUCROC/AUC-mix', num=14)
except:
    print("Not enough samples")

# %%


plt.figure(figsize=(20, 12))
sns.boxplot(x='classifier_name', y='accuracy', data=time_long, palette=colors)
sns.stripplot(x='classifier_name', y='accuracy', data=time_long, 
              size=10, color='.3', linewidth=0, alpha=0.6)

import matplotlib.patches as mpatches
classifiers = time_long['classifier_name'].unique()
patches = [mpatches.Patch(color=colors[i], label=classifier) for i, classifier in enumerate(classifiers)]
plt.legend(handles=patches, title='Classifier', loc='best', prop={'size': 25}, title_fontsize=20)

plt.ylabel('Time (Seconds)', fontsize=30)
plt.xlabel('', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
plt.savefig('results/transductive/Time/Time_Summary.png')
plt.show()
# %%

plt.figure(figsize=(20, 12))
time_long_sorted = time_long.sort_values(by='Observations')
# Order the colors according to the specified list
palette = {"ABOD": "#ec1763", "IForest": "#A25C43", "LOF": "#FABE37", "ECOD": "#91c059", "PCA": "#118D92", "Geary": "#204ecf"}
sns.scatterplot(x='Observations', y='time', hue='classifier_name', data=time_long_sorted,  s=500, alpha=0.6, palette=palette)

plt.ylabel('Time (Seconds)', fontsize=30)
plt.xlabel('Observations', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Create a custom legend to ensure the order
handles, labels = plt.gca().get_legend_handles_labels()
order = ["ABOD", "IForest", "LOF", "ECOD", "PCA", "Geary"]
ordered_handles = [handles[labels.index(label)] for label in order]
plt.legend(ordered_handles, order, title='Classifier', loc='best', prop={'size': 25}, title_fontsize=20)

plt.tight_layout()
plt.savefig('results/transductive/Time/Time_vs_Observations.png')
plt.show()
# %%

plt.figure(figsize=(20, 12))
time_long_sorted = time_long.sort_values(by='Observations')
# Order the colors according to the specified list
palette = {"ABOD": "#ec1763", "IForest": "#A25C43", "LOF": "#FABE37", "ECOD": "#91c059", "PCA": "#118D92", "Geary": "#204ecf"}
sns.scatterplot(x='Observations', y='time', hue='classifier_name', data=time_long_sorted,  s=500, alpha=0.6, palette=palette)

plt.ylabel('Time (Seconds)', fontsize=30)
plt.xlabel('Observations', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Create a custom legend to ensure the order
handles, labels = plt.gca().get_legend_handles_labels()
order = ["ABOD", "IForest", "LOF", "ECOD", "PCA", "Geary"]
ordered_handles = [handles[labels.index(label)] for label in order]
plt.legend(ordered_handles, order, title='Classifier', loc='best', prop={'size': 25}, title_fontsize=20)

plt.tight_layout()
plt.savefig('results/transductive/Time/Time_vs_Observations.png')
plt.show()
