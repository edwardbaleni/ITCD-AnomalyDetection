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

def getData():
    data = joblib.load("results/training/data0_70.pkl")
    path = "results/hyperparameter/"
    home = listdir(path)

    abod_files = [f'{path}{file}' for file in home if "ABOD" in file]
    lof_files = [f'{path}{file}' for file in home if "LOF" in file]
    pca_files = [f'{path}{file}' for file in home if "PCA" in file]
    if_files = [f'{path}{file}' for file in home if "IF" in file]

    return data, [joblib.load(file) for file in lof_files], [joblib.load(file) for file in abod_files], [joblib.load(file) for file in pca_files], [joblib.load(file) for file in if_files]


def process(data, erf_num, lof_params, abod_params, pca_params, if_params):

    return transductive.transductionResults(data, 
                                               str(erf_num), 
                                               lof_params, 
                                               abod_params, 
                                               pca_params, 
                                               if_params)

# if __name__ == "__main__":
    # Get sample size from user
# %%
# data, LOF, ABOD, PCA, IF = getData()
# sampleSize = len(LOF)
# # I have 20 cores!
# # with Pool(cpu_count() - 14) as pool:
# #     args = zip(data, LOF, ABOD, PCA, IF)
# #     results = pool.starmap(process, list(args))

# LOF[0].best_params
# aucroc = []
# ap = []
# time = []

# aucroc = pd.DataFrame()
# ap = pd.DataFrame()
# time = pd.DataFrame()

# for i in range(sampleSize):
#     y = np.array(data[i].loc[:, "Y"]).T 
#     y = np.where(y == 'Outlier', 1, 0)
#     if np.count_nonzero(y) == 0:
#         continue
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

# for i in range(len(aucroc)):
#     results_auc = pd.concat([results_auc, aucroc[i]], axis=0)
#     results_ap = pd.concat([results_ap, ap[i]], axis=0)
#     results_time = pd.concat([results_time, time[i]], axis=0)

# This is to separate the results
# TODO: The last result is std. deviation for each classifier over each orchard

# %%
# Save aucroc with only unique indices
aucroc = pd.read_csv("results/transductive/auc.csv")
ap = pd.read_csv("results/transductive/ap.csv")
time = pd.read_csv("results/transductive/time.csv")
aucroc_df = aucroc
ap_df = ap
time_df = time

# Save only unique aucroc indices
aucroc = aucroc.drop_duplicates(subset=['Data'])

# standard deviations can be found in pickle files
# time_df.to_csv('results/transductive/time.csv', index=False, float_format='%.3f')
# aucroc.to_csv('results/transductive/auc.csv', index=False, float_format='%.3f')
# ap_df.to_csv('results/transductive/ap.csv', index=False, float_format='%.3f')

# These dataframes need to be in long format for the CD diagram
aucroc_long = pd.melt(aucroc_df, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
ap_long = pd.melt(ap_df, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])
time_long = pd.melt(time_df,
                    id_vars=["Data"], 
                    value_vars= ['ABOD', 'IForest','LOF', 'ECOD', 'PCA', 'Geary'])

# Plot the CD diagram
from utils import cdDiagram

aucroc_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
ap_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
time_long.columns = ['dataset_name', 'classifier_name', 'accuracy']

aucroc_long = aucroc_long[['classifier_name', 'dataset_name', 'accuracy']]
ap_long = ap_long[['classifier_name', 'dataset_name', 'accuracy']]
time_long = time_long[['classifier_name', 'dataset_name', 'accuracy']]


cdDiagram.draw_cd_diagram(df_perf=ap_long, title='Average Precision', labels=True, measure = '/transductive/AP/')

cdDiagram.draw_cd_diagram(df_perf=aucroc_long, title='AUC-ROC', labels=True, measure = '/transductive/AUCROC/')

cdDiagram.draw_cd_diagram(df_perf=time_long, title='Time', labels=True, measure = '/transductive/Time/')
# %%
