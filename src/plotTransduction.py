# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

AUCROC_1, AUC_1, labels, P_1,R_1, AP_1 = pickle.load(open("results/transductive/0.pkl", "rb"))
AUCROC_2, AUC_2, labels, P_2,R_2, AP_2 = pickle.load(open("results/transductive/1.pkl", "rb"))
AUCROC_3, AUC_3, labels, P_3,R_3, AP_3 = pickle.load(open("results/transductive/2.pkl", "rb"))
AUCROC_4, AUC_4, labels, P_4,R_4, AP_4 = pickle.load(open("results/transductive/3.pkl", "rb"))
AUCROC_5, AUC_5, labels, P_5,R_5, AP_5 = pickle.load(open("results/transductive/4.pkl", "rb"))


AUCROC_1['Orchard'] = 'Orchard-1'
AUCROC_2['Orchard'] = 'Orchard-2'
AUCROC_3['Orchard'] = 'Orchard-3'
AUCROC_4['Orchard'] = 'Orchard-4'
AUCROC_5['Orchard'] = 'Orchard-5'

AUCROC = pd.concat([AUCROC_1, AUCROC_2, AUCROC_3, AUCROC_4, AUCROC_5])

AUC_1["Orchard"] = "Orchard-1"
AUC_2["Orchard"] = "Orchard-2"
AUC_3["Orchard"] = "Orchard-3"
AUC_4["Orchard"] = "Orchard-4"
AUC_5["Orchard"] = "Orchard-5"

AUCROC_auc = pd.concat([AUC_1, AUC_2, AUC_3, AUC_4, AUC_5])

# %%
plt.style.use('seaborn-v0_8-darkgrid')
palette = sns.color_palette("tab20", n_colors=12)
g = sns.relplot(
    data=AUCROC,
    x="FPR", y="TPR",
    hue="Estimator", style="Estimator", col="Type", row="Orchard",
    kind="line", palette=palette,
    height=5, aspect=1, facet_kws=dict(sharex=True, sharey=True), linewidth=5, zorder=5
)

# Increase label sizes for all axes
for ax in g.axes.flat:
    ax.set_xlabel('False-Positive Rate', fontsize=20)
    ax.set_ylabel('True-Positive Rate', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)

# Remove legend title and increase legend text size
g._legend.set_title(None)
for text in g._legend.get_texts():
    text.set_fontsize(30)
    
# Move and resize the legend
g._legend.set_bbox_to_anchor((1.2, 0.5))

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
            ax.text(0.6, y_pos, f"AUC: {row['AUC']}", transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='top', color=color)
            y_pos += 0.05
            
    # Set the title of each axis to the shared learn_approach
    ax.set_title("")
    
    # Set Headers nicely!
    ax.annotate(orchard_number, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                ha='right', va='center', fontsize=20, fontweight='heavy', rotation=90)  # Increased from 14 to 20

    
cols = AUCROC["Type"].unique()
axes = g.axes

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=20, fontweight='heavy', ha='center', va='baseline')

# Move the legend more to the right
g._legend.set_bbox_to_anchor((1.05, 0.5))
plt.savefig("results/transductive/AUCROC/ROC-Curve.png", dpi=300, bbox_inches='tight')

plt.show()




# %%
# TODO: Need to plot precision-recall curves for each estimator in each orchard

hold = {'Probabilistic':[], 'Learning':[], 'Distance':[], 'Density':[], 'Reconstruction':[], 'Spatial':[]}

for i,j in P_1.items():
    if i == 'ECOD':
        hold['Probabilistic'].append(i)
    elif i == 'ABOD':
        hold['Distance'].append(i)
    elif i == 'PCA':
        hold['Reconstruction'].append(i)
    elif i == 'IF':
        hold['Learning'].append(i)
    elif i == 'Geary':
        hold['Spatial'].append(i)
    else:
        hold['Density'].append(i)



# Orchard-110203, Orchard-111702
Orch = {0:[P_1, R_1, AP_1], 1:[P_2, R_2, AP_2], 2:[P_3, R_3, AP_3], 3:[P_4, R_4, AP_4], 4:[P_5, R_5, AP_5]}
Orch_index = {0:"Orchard-1", 1:"Orchard-2", 2:"Orchard-3", 3:"Orchard-4", 4:"Orchard-5"}

# %%
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(5, 6, figsize=(40, 25), sharex=True, sharey=True)

palette = sns.color_palette("tab20", n_colors=12)

# Define x-axis ticks
x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

for i, orchard in Orch.items():
    for type, est in hold.items():
        for estimator in est:
            j = list(hold.keys()).index(type)
            data = pd.DataFrame({"Precision": orchard[0][estimator], "Recall": orchard[1][estimator]})
            color = palette[AUCROC['Estimator'].unique().tolist().index(estimator)]
            ax[i, j].plot(data['Recall'], data['Precision'], label=f"AP: {orchard[2][estimator]}", color=color, linewidth=5)
            
            # Set x-axis ticks for all plots but only show labels on bottom row
            ax[i, j].set_xticks(x_ticks)
            if i == 4:  # Only show tick labels on bottom row
                ax[i, j].set_xticklabels(x_ticks, fontsize=18)
            else:
                ax[i, j].set_xticklabels([])
            
            # Only show y-ticks and y-labels on the first column
            if j == 0:
                ax[i, j].set_ylabel('Precision', fontsize=22)
                ax[i, j].set_yticks(y_ticks)
                ax[i, j].set_yticklabels(y_ticks, fontsize=18)
            else:
                ax[i, j].set_ylabel('')
                ax[i, j].tick_params(axis='y', which='both', left=True, labelleft=False)
                
            if i == 4:  # Only show xlabel for the bottom row
                ax[i, j].set_xlabel('Recall', fontsize=22)
            else:
                ax[i, j].set_xlabel('')
            ax[i, j].legend(fontsize=20)
            ax[i, j].axhline(y=0.5, linestyle='--', color='grey')

# Configure tick visibility
for i in range(5):
    for j in range(6):
        if i < 4:  # For all rows except the bottom
            ax[i, j].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
        else:  # For bottom row
            ax[i, j].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        
        # Increase tick font sizes for both axes
        ax[i, j].tick_params(axis='both', labelsize=18)

cols = list(hold.keys())
rows = list(Orch_index.values())

for axes, col in zip(ax[0], cols):
    axes.set_title(col, fontsize=26, fontweight='heavy')

pad = 5

for axes, row in zip(ax[:, 0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - pad, 0),
                  xycoords=axes.yaxis.label, textcoords='offset points',
                  ha='right', va='center', fontsize=26, fontweight='heavy', rotation=90)

# Add a legend for the estimator corresponding to each color with increased fontsize
handles = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(AUCROC['Estimator'].unique()))]
labels = AUCROC['Estimator'].unique()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=30)

fig.tight_layout()
plt.savefig("results/transductive/AP/PR-Curve.png", dpi=300, bbox_inches='tight')

plt.show()




# %%

# TODO: Plot orchards labels
#       This will be used in the final analysis to demonstrate the performance of each estimator in each orchard
#       Simply use plotAnomaly code!
#       Can also plot the decision scores!

#       Choose to plot the 5 orchards shown here
