
# %%
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

df_1, df_auc_1 = pickle.load(open("results/inductive/110021.pkl", "rb"))
df_2, df_auc_2 = pickle.load(open("results/inductive/114348.pkl", "rb"))
df_3, df_auc_3 = pickle.load(open("results/inductive/122075.pkl", "rb"))
df_4, df_auc_4 = pickle.load(open("results/inductive/124018.pkl", "rb"))

df_1['Orchard'] = '110021'
df_2['Orchard'] = '114348'
df_3['Orchard'] = '122075'
df_4['Orchard'] = '124018'

df = pd.concat([df_1, df_2, df_3, df_4])

df_auc_1["Orchard"] = "110021"
df_auc_2["Orchard"] = "114348"
df_auc_3["Orchard"] = "122075"
df_auc_4["Orchard"] = "124018"

df_auc = pd.concat([df_auc_1, df_auc_2, df_auc_3, df_auc_4])





palette = sns.color_palette("tab20", n_colors=11)

g = sns.relplot(
    data=df,
    x="FPR", y="TPR",
    hue="Estimator", col="Type", row="Orchard",
    kind="line", palette=palette,
    height=5, aspect=1, facet_kws=dict(sharex=True), linewidth=5, zorder=5
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
        # if 
        if row['Estimator'] in df[df['Type'] == learn_approach]['Estimator'].unique() and row['Orchard'] == orchard_number:
            color = palette[df['Estimator'].unique().tolist().index(row['Estimator'])]
            ax.text(0.6, y_pos, f"AUC: {row['AUC']} ± {row['std']}", transform=ax.transAxes, fontsize=12, verticalalignment='top', color=color)
            y_pos += 0.05
    ax.title("")
    # Set Headers nicely!
    ax.annotate(learn_approach,xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
    ax.annotate(orchard_number, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

plt.show()

# %%
