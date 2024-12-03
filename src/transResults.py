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


def getDataNames(sampleSize = 70):
    return utils.collectFiles(sampleSize)

def process(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
    # Your code goes here
    # num = 0
    # Do not scale data here because it will be split into training and testing data
    myData = utils.engineer(0,
                            [data_paths_tif], 
                            [data_paths_geojson], 
                            [data_paths_geojson_zipped],
                            False)
    
    data = myData.data.copy(deep=True)
    erf_num = myData.erf

    results = transductive.transductionResults(data, erf_num)
    
    return results

# %%
if __name__ == "__main__":
    # Get sample size from user
    sampleSize = 2

    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = getDataNames(sampleSize)
    
    # I have 20 cores!
    with Pool(cpu_count() - 18) as pool:
        args = zip(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
        results = pool.starmap(process, list(args))

    # This is to separate the results
    # TODO: The last result is std. deviation for each classifier over each orchard

    auroc, ap, time, std_auc, std_auc = zip(*results)

    auroc_df = pd.DataFrame()
    ap_df = pd.DataFrame()
    time_df = pd.DataFrame()
    # turn these tuples into dataframes
    for i in range(sampleSize):
        auroc_df = pd.concat([auroc_df, auroc[i]], axis=0)
        ap_df = pd.concat([ap_df, ap[i]], axis=0)
        time_df = pd.concat([time_df, time[i]], axis=0)

    # standard deviations can be found in pickle files
    time_df.to_csv('results/inductive/time.csv', index=False, float_format='%.3f')
    auroc_df.to_csv('results/inductive/auc.csv', index=False, float_format='%.3f')
    ap_df.to_csv('results/inductive/ap.csv', index=False, float_format='%.3f')

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

    try:
        cdDiagram.draw_cd_diagram(df_perf=ap_long, title='Average Precision', labels=True, measure = 'AP')
    except:
        print("Error in AP")


    try:
        cdDiagram.draw_cd_diagram(df_perf=auroc_long, title='Average Precision', labels=True, measure = 'AUCROC')
    except:
        print("Error in AUCROC")