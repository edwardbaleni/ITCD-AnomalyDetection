# https://www.statology.org/plot-roc-curve-python/
# https://stackoverflow.com/questions/66505014/how-to-add-auc-to-a-multiple-roc-graph-with-procs-ggroc
# https://www.geeksforgeeks.org/working-with-multiple-plots-faceting/
# https://stackoverflow.com/questions/67810248/in-ggplot2-specify-a-confidence-interval-95-ci-around-geom-smooth-or-any-tr
# %%
import pickle

favorite_color = pickle.load(open("results/inductive/110021.pkl", "rb"))


# %%
import pandas as pd

tprs = pd.DataFrame(favorite_color[0])
fprs = pd.DataFrame(favorite_color[1])
aucs = favorite_color[2]
tpr_std = pd.DataFrame(favorite_color[3])
tpr_upper = pd.DataFrame(favorite_color[4])
tpr_lower = pd.DataFrame(favorite_color[5])
std_auc = favorite_color[6]

# %%
tpr_df = tprs.melt(var_name='Estimator', value_name='TPR')

fpr_df = fprs.melt(var_name='Estimator', value_name='FPR')

tpr_upper_df = tpr_upper.melt(var_name='Estimator', value_name='TPR_Upper')

tpr_lower_df = tpr_lower.melt(var_name='Estimator', value_name='TPR_Lower')

df = pd.concat([tpr_df, fpr_df, tpr_upper_df, tpr_lower_df], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

df['Type'] = None

for i in df["Estimator"].unique():
    if i == 'ABOD' or i == 'COPOD' or i == 'ECOD' or i == 'HBOS':
        df.loc[df['Estimator'] == i, 'Type'] = 'Probabilistic'
    elif i == 'CBLOF':
        df.loc[df['Estimator'] == i, 'Type'] = 'Cluster'
    elif i == 'IF' or i == 'KNN':
        df.loc[df['Estimator'] == i, 'Type'] = 'Distance'
    elif i == 'KPCA':
        df.loc[df['Estimator'] == i, 'Type'] = 'Reconstruction'
    else:
        df.loc[df['Estimator'] == i, 'Type'] = 'Density'

# %%
from plotnine import ggplot, aes, geom_line, geom_ribbon, facet_wrap, facet_grid,theme, theme_minimal, labs

# instead use facet grid, of type against orchard
(
    ggplot(df, aes(x='FPR', y='TPR', color='Estimator')) +
    geom_ribbon(aes(ymin='TPR_Lower', ymax='TPR_Upper'), alpha=0.5) +
    geom_line() +
    facet_wrap('Type', ncol=5) +
    labs(title='ROC Curve', x='False Positive Rate', y='True Positive Rate') +
    theme_minimal()
)


