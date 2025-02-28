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

    results = transductive.transductionResults(data, 
                                               str(erf_num), 
                                               lof_params, 
                                               abod_params, 
                                               pca_params, 
                                               if_params)
    
    return results

# if __name__ == "__main__":
    # Get sample size from user
# %%
data, LOF, params, PCA, IF = getData()
sampleSize = 14#len(data)
# I have 20 cores!
# with Pool(cpu_count() - 14) as pool:
#     args = zip(data, LOF, ABOD, PCA, IF)
#     results = pool.starmap(process, list(args))

from pyod.models.abod import ABOD
a = ABOD( params[0].best_params )

for i in range(sampleSize):
    aucroc, ap, time = process(data[i], 
                      i, 
                      LOF[i].best_params, 
                      ABOD[i].best_params, 
                      PCA[i].best_params, 
                      IF[i].best_params)



# This is to separate the results
# TODO: The last result is std. deviation for each classifier over each orchard

# %%
auroc, ap, time = zip(*results)

auroc_df = pd.DataFrame()
ap_df = pd.DataFrame()
time_df = pd.DataFrame()
# turn these tuples into dataframes
for i in range(sampleSize):
    auroc_df = pd.concat([auroc_df, auroc[i]], axis=0)
    ap_df = pd.concat([ap_df, ap[i]], axis=0)
    time_df = pd.concat([time_df, time[i]], axis=0)

# standard deviations can be found in pickle files
time_df.to_csv('results/transductive/time.csv', index=False, float_format='%.3f')
auroc_df.to_csv('results/transductive/auc.csv', index=False, float_format='%.3f')
ap_df.to_csv('results/transductive/ap.csv', index=False, float_format='%.3f')

# These dataframes need to be in long format for the CD diagram

auroc_long = pd.melt(auroc_df, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD', 'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD'])
ap_long = pd.melt(ap_df, 
                        id_vars=["Data"], 
                        value_vars= ['ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD', 'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD'])
time_long = pd.melt(time_df,
                    id_vars=["Data"], 
                    value_vars= ['ABOD', 'CBLOF', 'HBOS', 'IForest', 'KNN', 'MCD', 'LOF', 'ECOD', 'KPCA', 'INNE', 'COPOD'])


# Plot the CD diagram
from utils import cdDiagram



auroc_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
ap_long.columns = ['dataset_name', 'classifier_name', 'accuracy']
time_long.columns = ['dataset_name', 'classifier_name', 'accuracy']

auroc_long = auroc_long[['classifier_name', 'dataset_name', 'accuracy']]
ap_long = ap_long[['classifier_name', 'dataset_name', 'accuracy']]
time_long = time_long[['classifier_name', 'dataset_name', 'accuracy']]

df_perf = auroc_long

# try:
#     cdDiagram.draw_cd_diagram(df_perf=ap_long, title='Average Precision', labels=True, measure = 'transductive/AP')
# except:
#     print("Error in AP")


# try:
#     cdDiagram.draw_cd_diagram(df_perf=auroc_long, title='Average Precision', labels=True, measure = 'transductive/AUCROC')
# except:
#     print("Error in AUCROC")