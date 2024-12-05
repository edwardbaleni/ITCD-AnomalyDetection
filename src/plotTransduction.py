# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

AUCROC_1, AUC_1, labels, P_1,R_1, AP_1 = pickle.load(open("results/transductive/110203.pkl", "rb"))
AUCROC_2, AUC_2, labels, P_2,R_2, AP_2 = pickle.load(open("results/transductive/111702.pkl", "rb"))


AUCROC_1['Orchard'] = 'Orchard-110203'
AUCROC_2['Orchard'] = 'Orchard-111702'

AUCROC = pd.concat([AUCROC_1, AUCROC_2])

AUC_1["Orchard"] = "Orchard-110203"
AUC_2["Orchard"] = "Orchard-111702"

AUCROC_auc = pd.concat([AUC_1, AUC_2])



plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("tab20", n_colors=12)
g = sns.relplot(
    data=AUCROC,
    x="FPR", y="TPR",
    hue="Estimator", style="Estimator", col="Type", row="Orchard",
    kind="line", palette=palette,
    height=5, aspect=1, facet_kws=dict(sharex=True, sharey=True), linewidth=5, zorder=5
)

pad = 5

for ax in g.axes.flat:
    # Go into each axis and add the chance line
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    # Obtain axis title, e.g. Prob., Recon., Clust. etc. 
    learn_approach = ax.get_title().split(' = ')[-1]
    # Obtain orchard number
    orchard_number = ax.get_title().split(' = ')[1].split(' ')[0]
    y_pos = 0.1
    # now we want to add an auc value for each estimator in each orchard
    for _, row in AUCROC_auc.iterrows():
        # if estimator and orchard exist in axis then add the auc value
        if row['Estimator'] in AUCROC[AUCROC['Type'] == learn_approach]['Estimator'].unique() and row['Orchard'] == orchard_number:
            color = palette[AUCROC['Estimator'].unique().tolist().index(row['Estimator'])]
            ax.text(0.6, y_pos, f"AUC: {row['AUC']}", transform=ax.transAxes, fontsize=12, verticalalignment='top', color=color)
            y_pos += 0.05
            
    # Set the title of each axis to the shared learn_approach
    ax.set_title("")
    
    # Set Headers nicely!
    ax.annotate(orchard_number, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                ha='right', va='center', fontsize=14, fontweight='heavy', rotation=90)

    
cols = AUCROC["Type"].unique()
axes = g.axes

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=14, fontweight='heavy', ha='center', va='baseline')

# Increase the size of the legend
g._legend.set_title(g._legend.get_title().get_text(), prop={'size': 14})
for text in g._legend.get_texts():
    text.set_fontsize(18)

# Move the legend more to the right
g._legend.set_bbox_to_anchor((1, 0.5))
    
plt.show()




# %%
# TODO: Need to plot precision-recall curves for each estimator in each orchard

hold = {'Probabilistic':[], 'Cluster':[], 'Distance':[], 'Density':[], 'Reconstruction':[], 'Spatial':[]}

for i,j in P_1.items():
    if i == 'ABOD' or i == 'COPOD' or i == 'ECOD' or i == 'HBOS':
        hold['Probabilistic'].append(i)
    elif i == 'CBLOF':
        hold['Cluster'].append(i)
    elif i == 'IF' or i == 'KNN':
        hold['Distance'].append(i)
    elif i == 'KPCA':
        hold['Reconstruction'].append(i)
    elif i == 'Geary':
        hold['Spatial'].append(i)
    else:
        hold['Density'].append(i)




# Orchard-110203, Orchard-111702
Orch = {0:[P_1, R_1, AP_1], 1:[P_2, R_2, AP_2]}
Orch_index = {0:"Orchard-110203", 1:"Orchard-111702"}

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(2, 6, figsize=(40, 15), sharex=True, sharey=True)

palette = sns.color_palette("tab20", n_colors=12)

for i, orchard in Orch.items():
    for type, est in hold.items():
        for estimator in est:
            j = list(hold.keys()).index(type)
            data = pd.DataFrame({"Precision": orchard[0][estimator], "Recall": orchard[1][estimator]})
            color = palette[AUCROC['Estimator'].unique().tolist().index(estimator)]
            ax[i, j].plot(data['Recall'], data['Precision'], label=f"AUC: {orchard[2][estimator]}", color=color, linewidth=5)
            ax[i, j].set_xlabel('Recall')
            ax[i, j].set_ylabel('Precision')
            ax[i, j].legend()
            ax[i, j].axhline(y=0.5, linestyle='--', color='grey')#, label='Chance')

cols = list(hold.keys())
rows = list(Orch_index.values())

for axes, col in zip(ax[0], cols):
    axes.set_title(col, fontsize=14, fontweight='heavy')

pad = 5

for axes, row in zip(ax[:, 0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - pad, 0),
                  xycoords=axes.yaxis.label, textcoords='offset points',
                  ha='right', va='center', fontsize=14, fontweight='heavy', rotation=90)

# Add a legend for the estimator corresponding to each color
handles = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(AUCROC['Estimator'].unique()))]
labels = AUCROC['Estimator'].unique()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

fig.tight_layout()#rect=[0, 0, 1, 0.95])

plt.show()




# %%

# TODO: Plot orchards labels
#       This will be used in the final analysis to demonstrate the performance of each estimator in each orchard
#       Simply use plotAnomaly code!
#       Can also plot the decision scores!

#       Choose to plot the 5 orchards shown here