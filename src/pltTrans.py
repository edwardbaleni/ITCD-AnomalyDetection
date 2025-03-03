# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

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


for i in range(0, 20):
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


# %%
plt.style.use('seaborn-v0_8-darkgrid')
# Define the desired estimator order
desired_order = ["ABOD", "IF", "LOF", "ECOD", "PCA", "Geary"]

# Create subplots
fig, axs = plt.subplots(4, 5, figsize=(30, 20))
orchards = roc['Orchard'].unique()

# Create a palette using the desired estimator order
palette = dict(zip(desired_order, sns.color_palette("rocket_r", len(desired_order))))

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
                linewidth=4)
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
plt.savefig("results/transductive/AUCROC/ROC-Curve.png", dpi=300, bbox_inches='tight')
plt.show()


# %%

pr = pr[~((pr["Precision"] == 0) & (pr["Recall"] == 0))]

plt.style.use('seaborn-v0_8-darkgrid')
# Define the desired estimator order
desired_order = ["ABOD", "IF", "LOF", "ECOD", "PCA", "Geary"]
# Create subplots
fig, axs = plt.subplots(4, 5, figsize=(30, 20))
orchards = pr['Orchard'].unique()
# Create a palette using the desired estimator order
palette = dict(zip(desired_order, sns.color_palette("rocket_r", len(desired_order))))
nrows, ncols = axs.shape

for i, (ax, orchard) in enumerate(zip(axs.flat, orchards)):
    data_orch = pr[pr['Orchard'] == orchard]
    # Only consider estimators that exist in the current orchard in the desired order
    estimators = [est for est in desired_order if est in data_orch['Estimator'].unique()]
    for estimator in estimators:
        data_est = data_orch[data_orch['Estimator'] == estimator]
        # Retrieve the corresponding AUC value from the auc dataframe
        ap_val_series = ap[(ap['Orchard'] == orchard) & (ap['Estimator'] == estimator)]['AP']
        if not ap_val_series.empty:
            ap_val = float(ap_val_series.iloc[0])
        else:
            ap_val = float('nan')
        ax.plot(data_est['Recall'], data_est['Precision'],
                label=f"{estimator}={ap_val:.3f}",
                color=palette.get(estimator, 'black'),
                linewidth=4)
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
plt.savefig("results/transductive/AP/PR-Curve.png", dpi=300, bbox_inches='tight')
plt.show()



# %%
