import joblib
import pandas as pd
import numpy as np
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from Model import Geary
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
import seaborn as sns

def anomaly(data, num, group):
    """Anomaly detection using the ABOD, ECOD, IF, LOF, and Geary models.
    Parameters:
    data (dict): A dictionary containing the training and testing data.
    Returns:
    dict: A dictionary containing the anomaly scores for each model.
    """
    y = np.array(data.loc[:, "Y"]).T 
    y = np.where(y == 'Outlier', 1, 0)
    outliers_fraction = np.count_nonzero(y) / len(y) if np.count_nonzero(y) > 0 else 0.01

    # move numerically instead of by column name
    # becuase conf will be removed at some point
    scaler = RobustScaler()
    X = data.iloc[:, 5:]
    predictors = X.columns
    X = scaler.fit_transform(X)

    geometry = data["geometry"]
    centroid = data["centroid"]

    models = {
        "ABOD": ABOD(contamination=outliers_fraction),
        "ECOD": ECOD(contamination=outliers_fraction),
        "IForest": IForest(contamination=outliers_fraction),
        "LOF": LOF(contamination=outliers_fraction),
        "PCA": PCA(contamination=outliers_fraction),
        "Geary": Geary(contamination=outliers_fraction, 
                       geometry=geometry, 
                       centroid=centroid)
    }

    auc = {"Orchard": f"Orchard {num + 1}", "Group": group}
    ap = {"Orchard": f"Orchard {num + 1}", "Group": group}

    for model_name, model in models.items():
        model.fit(X)
        auc[model_name] = roc_auc_score(y, model.decision_scores_) if len(np.unique(y)) > 1 else 0
        ap[model_name] = average_precision_score(y, model.decision_scores_)

    return (pd.DataFrame.from_dict(auc, orient='index').T, pd.DataFrame.from_dict(ap, orient='index').T)

if __name__ == "__main__":
    data = joblib.load("results/training/datafull0_70.pkl")

    # using the feature set groupings, test the performance of each group
    # on default settings on each model!

    #    All
    df_columns = ['Orchard', 'Group', 'ABOD', 'ECOD', 'IForest', 'LOF', 'PCA', 'Geary']
    output_auc = pd.DataFrame(columns=df_columns)
    output_ap = pd.DataFrame(columns=df_columns)
    rng = len(data)
    for i in range(rng):
        auc, ap = anomaly(data[i], i, "All")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)

    #    No Confidence
    data_conf = data[:]

    for i in range(rng):
        data_conf[i] = data_conf[i].drop(columns=['confidence'])
        auc, ap = anomaly(data_conf[i], i, "No Confidence")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)
        
    #    No Shape
    data_shape = data[:]

    for i in range(rng):
        data_shape[i] = data_shape[i].drop(columns=['roundness', 
                                        'compactness', 
                                        'convexity', 
                                        'solidity',
                                        'bendingE'])
        auc, ap = anomaly(data_shape[i], i, "No Shape")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)

    #    No Zernike
    data_zern = data[:]

    for i in range(rng):
        data_zern[i] = data_zern[i].drop(columns=['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
        'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17',
        'z18', 'z19', 'z20', 'z21', 'z22', 'z23', 'z24'])
        auc, ap = anomaly(data_zern[i], i, "No Zernike")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)

    #    No Spec
    data_spec = data[:]

    for i in range(rng):
        data_spec[i] = data_spec[i].drop(columns=['NDRE',
                                                  'NDVI', 
                                                  'EVI', 
                                                  'OSAVI'])
        auc, ap = anomaly(data_spec[i], i, "No Spectral")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)

    #    No Haralick
    data_text = data[:]

    for i in range(rng):
        data_text[i] = data_text[i].drop(columns=['Corr','ASM'])
        auc, ap = anomaly(data_text[i], i, "No Texture")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)

    #    No DSM
    data_dsm = data[:]

    for i in range(rng):
        data_dsm[i] = data_dsm[i].drop(columns=['DSM'])
        auc, ap = anomaly(data_dsm[i], i, "No DSM")
        output_auc = pd.concat([output_auc, auc], axis=0)
        output_ap = pd.concat([output_ap, ap], axis=0)


    output_ap.reset_index(drop=True, inplace=True)
    output_auc.reset_index(drop=True, inplace=True)

    joblib.dump(output_auc, "results/ablation/auc.pkl")
    joblib.dump(output_ap, "results/ablation/ap.pkl")

    output_melted = output_auc.melt(id_vars=['Orchard', 'Group'], var_name='Model', value_name='auc')    
    # output_melted['Model'] = output_melted['Model'].replace('IF', 'IForest')
    # output_melted = output_melted[~((output_melted['Model'] == 'IForest') & (output_melted['auc'].isna()))]

    # order = ['ABOD', 'ECOD', 'IForest', 'LOF', 'PCA', 'Geary']

    # output_melted['Model'] = pd.Categorical(output_melted['Model'], categories=order, ordered=True)
    plt.figure(figsize=(20, 12))
    sns.boxplot(x='Model', y='auc', hue='Group', data=output_melted)
    # plt.title('Anomaly Detection Model Performance by Feature Group')
    plt.xlabel('Model', fontsize=30)
    plt.ylabel('ROC-AUC Score', fontsize=30)
    plt.legend(title='Feature Group')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig('results/ablation/AUC.png')
    plt.show()

    output_melted = output_ap.melt(id_vars=['Orchard', 'Group'], var_name='Model', value_name='ap')
    # output_melted['Model'] = output_melted['Model'].replace('IF', 'IForest')
    # output_melted = output_melted[~((output_melted['Model'] == 'IForest') & (output_melted['ap'].isna()))]


    # order = ['ABOD', 'ECOD', 'IForest', 'LOF', 'PCA', 'Geary']

    # output_melted['Model'] = pd.Categorical(output_melted['Model'], categories=order, ordered=True)

    plt.figure(figsize=(20, 12))
    sns.boxplot(x='Model', y='ap', hue='Group', data=output_melted)
    # plt.title('Anomaly Detection Model Performance by Feature Group')
    plt.xlabel('Model', fontsize=30)
    plt.ylabel('Average Precision', fontsize=30)
    plt.legend(title='Feature Group')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig('results/ablation/AP.png')
    plt.show()

    