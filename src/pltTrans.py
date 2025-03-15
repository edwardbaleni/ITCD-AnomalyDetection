# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib
import numpy as np

def PrecisionRecall(P_1, R_1, AP,Orchard):

    PR = pd.DataFrame()

    for i,j in zip(P_1.items(), R_1.items()):
        newPR = pd.DataFrame({"Precision": i[1], "Recall": j[1]})
        newPR['Estimator'] = i[0]
        PR = pd.concat([PR, newPR], axis=0)
    
    PR['Orchard'] = Orchard

    AP = pd.DataFrame.from_dict(AP, orient='index')
    AP.rename(columns={0: 'AP'}, inplace=True)
    AP['Estimator'] = AP.index
    AP['Orchard'] = Orchard
    AP.reset_index(drop=True, inplace=True)

    return PR, AP

roc = pd.DataFrame()
auc = pd.DataFrame()

pr = pd.DataFrame()
ap = pd.DataFrame()


for i in range(0, 9):
    AUCROC, AUC, labels, P, R, AP = pickle.load(open(f"results/transductive/{i}.pkl", "rb"))

    AUCROC['Orchard'] = f'Orchard {i+1}'
    AUC['Orchard'] = f'Orchard {i+1}'
    roc = pd.concat([roc, AUCROC])
    auc = pd.concat([auc, AUC])
    # precision_recall = pd.DataFrame()
    precision_recall, AP = PrecisionRecall(P, R, AP, f"Orchard {i+1}")

    pr = pd.concat([pr, precision_recall])
    ap = pd.concat([ap, AP])

roc.reset_index(drop=True, inplace=True)
auc.reset_index(drop=True, inplace=True)

# Replace "IF" with "IForest" in roc and auc dataframes
roc['Estimator'] = roc['Estimator'].replace('IF', 'IForest')
auc['Estimator'] = auc['Estimator'].replace('IF', 'IForest')

# %%
plt.style.use('seaborn-v0_8-darkgrid')
# Define the desired estimator order
desired_order = ["ABOD", "IForest", "LOF", "ECOD", "PCA", "Geary"]

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(30, 20))
orchards = roc['Orchard'].unique()

# Create a palette using the desired estimator order
        # Yellow,     pink,      teal,      lime,     dark blue,   brown
# color =  ["#FABE37", "#ec1763", "#118D92", "#91c059", "#204ecf", "#A25C43"]
        # pink       brown      yellow      lime       teal     blue
color = ["#ec1763", "#A25C43", "#FABE37", "#91c059", "#118D92", "#204ecf"]
palette = dict(zip(desired_order, sns.color_palette(sns.set_palette(color), len(desired_order))))
# Accent
# palette = dict(zip(desired_order, sns.color_palette("nipy_spectral", len(desired_order))))

nrows, ncols = axs.shape

for i, (ax, orchard) in enumerate(zip(axs.flat, orchards)):
    data_orch = roc[roc['Orchard'] == orchard]
    # Only consider estimators that exist in the current orchard in the desired order
    estimators = [est for est in desired_order if est in data_orch['Estimator'].unique()]
    for estimator in estimators:
        data_est = data_orch[data_orch['Estimator'] == estimator]
        # Retrieve the corresponding AUC value from the auc dataframe
        auc_val_series = auc[(auc['Orchard'] == orchard) & (auc['Estimator'] == estimator)]['AUC']
        if not auc_val_series.empty:
            auc_val = float(auc_val_series.iloc[0])
        else:
            auc_val = float('nan')
        ax.plot(data_est['FPR'], data_est['TPR'],
                label=f"{estimator}={auc_val:.3f}",
                color=palette.get(estimator, 'black'),
                linewidth=4, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title(orchard, fontsize=24)
    ax.legend(title="AUC", loc="lower right", fontsize=16, title_fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    
    # Hide ylabels and yticks for columns other than the first
    col_index = i % ncols
    if col_index == 0:
        ax.set_ylabel("True Positive Rate", fontsize=24)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    
    # Only set xticks and xlabels for the last row
    row_index = i // ncols
    if row_index == nrows - 1:
        ax.set_xlabel("False Positive Rate", fontsize=24)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

plt.tight_layout()
plt.savefig("results/transductive/AUCROC/ROC-Curve_subset.png", dpi=300, bbox_inches='tight')
plt.show()


# %%

data = joblib.load("results/training/data0_70.pkl")


pr = pr[~((pr["Precision"] == 0) & (pr["Recall"] == 0))]

# Replace "IF" with "IForest" in roc and auc dataframes
pr['Estimator'] = pr['Estimator'].replace('IF', 'IForest')
ap['Estimator'] = ap['Estimator'].replace('IF', 'IForest')

plt.style.use('seaborn-v0_8-darkgrid')
# Define the desired estimator order
desired_order = ["ABOD", "IForest", "LOF", "ECOD", "PCA", "Geary"]
# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(30, 20))
orchards = pr['Orchard'].unique()
# Create a palette using the desired estimator order

# palette = dict(zip(desired_order, sns.color_palette("rocket_r", len(desired_order))))
        # pink       brown      yellow      lime       teal     blue
color = ["#ec1763", "#A25C43", "#FABE37", "#91c059", "#118D92", "#204ecf"]
palette = dict(zip(desired_order, sns.color_palette(sns.set_palette(color), len(desired_order))))
nrows, ncols = axs.shape

for i, (ax, orchard) in enumerate(zip(axs.flat, orchards)):
    data_orch = pr[pr['Orchard'] == orchard]
    # Only consider estimators that exist in the current orchard in the desired order
    estimators = [est for est in desired_order if est in data_orch['Estimator'].unique()]

    # Obtain the baseline precision and recall values
    y = np.array(data[i].loc[:, "Y"]).T 
    y = np.where(y == 'Outlier', 1, 0)
    
    outliers_fraction = np.count_nonzero(y) / len(y) 

    for estimator in estimators:
        data_est = data_orch[data_orch['Estimator'] == estimator]
        # Retrieve the corresponding AP value from the ap dataframe
        ap_val_series = ap[(ap['Orchard'] == orchard) & (ap['Estimator'] == estimator)]['AP']
        if not ap_val_series.empty:
            ap_val = float(ap_val_series.iloc[0])
        else:
            ap_val = float('nan')
        ax.plot(data_est['Recall'], data_est['Precision'],
                label=f"{estimator}={ap_val:.3f}",
                color=palette.get(estimator, 'black'),
                linewidth=4, alpha=0.8)
    # Draw a horizontal line at Precision = 0.5
    ax.axhline(y=outliers_fraction, color='gray', linestyle='--', linewidth=2)
    ax.set_title(orchard, fontsize=24)
    ax.legend(title="AP", loc="upper right", fontsize=16, title_fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    
    # Hide ylabels and yticks for columns other than the first
    col_index = i % ncols
    if col_index == 0:
        ax.set_ylabel("Precision", fontsize=24)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    
    # Only set xticks and xlabels for the last row
    row_index = i // ncols
    if row_index == nrows - 1:
        ax.set_xlabel("Recall", fontsize=24)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
        
plt.tight_layout()
plt.savefig("results/transductive/AP/PR-Curve_subset.png", dpi=300, bbox_inches='tight')
plt.show()



# %%


