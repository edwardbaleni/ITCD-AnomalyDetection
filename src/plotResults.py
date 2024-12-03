# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

df_1, df_auc_1 = pickle.load(open("results/inductive/110021.pkl", "rb"))
df_2, df_auc_2 = pickle.load(open("results/inductive/114348.pkl", "rb"))
df_3, df_auc_3 = pickle.load(open("results/inductive/122075.pkl", "rb"))
df_4, df_auc_4 = pickle.load(open("results/inductive/124018.pkl", "rb"))

df_1['Orchard'] = 'Orchard-110021'
df_2['Orchard'] = 'Orchard-114348'
df_3['Orchard'] = 'Orchard-122075'
df_4['Orchard'] = 'Orchard-124018'

df = pd.concat([df_1, df_2, df_3, df_4])

df_auc_1["Orchard"] = "Orchard-110021"
df_auc_2["Orchard"] = "Orchard-114348"
df_auc_3["Orchard"] = "Orchard-122075"
df_auc_4["Orchard"] = "Orchard-124018"

df_auc = pd.concat([df_auc_1, df_auc_2, df_auc_3, df_auc_4])

palette = sns.color_palette("tab20", n_colors=11)
g = sns.relplot(
    data=df,
    x="FPR", y="TPR",
    hue="Estimator", style="Estimator", col="Type", row="Orchard",
    kind="line", palette=palette,
    height=5, aspect=1, facet_kws=dict(sharex=True, sharey=True), linewidth=5, zorder=5
)


# Add chance line
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
    for _, row in df_auc.iterrows():
        # if estimator and orchard exist in axis then add the auc value
        if row['Estimator'] in df[df['Type'] == learn_approach]['Estimator'].unique() and row['Orchard'] == orchard_number:
            color = palette[df['Estimator'].unique().tolist().index(row['Estimator'])]
            ax.text(0.6, y_pos, f"AUC: {row['AUC']} ± {row['std']}", transform=ax.transAxes, fontsize=12, verticalalignment='top', color=color)
            y_pos += 0.05
            # Add confidence intervals (not really CI but the range of the TPR)
            ax.fill_between(df[(df['Estimator'] == row['Estimator']) & (df['Orchard'] == row['Orchard']) ]['FPR'], 
                            df[(df['Estimator'] == row['Estimator']) & (df['Orchard'] == row['Orchard']) ]['TPR_Lower'], 
                            df[(df['Estimator'] == row['Estimator']) & (df['Orchard'] == row['Orchard']) ]['TPR_Upper'], 
                            alpha=0.2)
    
    # Set the title of each axis to the shared learn_approach
    ax.set_title("")
    
    # Set Headers nicely!
    ax.annotate(orchard_number, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                ha='right', va='center', fontsize=14, fontweight='heavy', rotation=90)

    
cols = df["Type"].unique()
axes = g.axes

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=14, fontweight='heavy', ha='center', va='baseline')
    
plt.show()


