# %%
import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import utils.plotAnomaly as plot
import plotly.express as px
import utils.Triangulation as tri

import joblib

from Model import transductive

# %%
# plot boxplots for each group and separate by orchard
def box_plot_comparison(data, feature_group=None):
    # Filter by feature group if specified

    # Create box plots for each feature grouped by orchard
    plt.figure(figsize=(20, 12))
    sns.boxplot(x='Feature', y='Value', hue='Orchard', data=data)
    plt.xticks(fontsize=25, rotation=90)
    plt.yticks(fontsize=25)
    # plt.title(f'Box Plot of Feature Values by Orchard ({feature_group or "All Groups"})')
    plt.ylabel('Value', fontsize=30)
    plt.xlabel('')
    plt.legend(loc='upper right', fontsize=25)
    plt.tight_layout()
    plt.savefig("results/EDA/Boxplots/{}.png".format(feature_group))


def getDataNames(sampleSize):
    return utils.collectFiles(sampleSize)

# def process(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
#     # Your code goes here
#     # num = 0
#     # Do not scale data here because it will be split into training and testing data
#     myData = utils.engineer(0,
#                             [data_paths_tif], 
#                             [data_paths_geojson], 
#                             [data_paths_geojson_zipped],
#                             False) # False)
    
#     data = myData.data.copy(deep=True)
#     mask = myData.mask
#     spectralData = myData.spectralData

#     # For plotting
#     img = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
#     img = img/255
    
#     return data, img

# if __name__ == '__main__':
    # Get sample size from user
    # sampleSize = 5

    # data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = getDataNames(sampleSize)

    # # I have 20 cores!
    # with Pool(5) as pool:
    #     args = zip(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
    #     results = pool.starmap(process, list(args))

    # dataOriginal, img = zip(*results)
    # data = list(dataOriginal)[:]

# %%
data = joblib.load("results/training/data0_12.pkl")
sampleSize = 5

# %%
dataOriginal, spec, masks = data['data'][0:5], data['spectralData'][0:5], data['mask'][0:5]  

# TODO: Do this for entire dataset and save separately
columns_to_remove = ["minor_axis", "radius_of_gyration", "major_axis"]
for i in range(sampleSize):
    dataOriginal[i].drop(columns=columns_to_remove, inplace=True)

data = list(dataOriginal)[:]
img = []

for i in range(5):
    mask = masks[i]
    # For plotting
    img.append(spec[i]["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)/255)


# Label orchards according to orchard
for i in range(sampleSize):
    data[i]["Orchard"] = "Orchard {}".format(i+1)
    data[i] = data[i].loc[:, ["Orchard"] + list(data[i].columns[:-1])]

# Remove specified columns from data # nir, red, reg, green and blue have already been removed
columns_to_remove = ["GNDVI", "NIR"]
for i in range(sampleSize):
    data[i].drop(columns=columns_to_remove, inplace=True)

# %%
# Group data by groups
spec = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "DSM":"OSAVI"].columns)] for i in range(sampleSize)])
contrast = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "Contrast":"Contrast"].columns)] for i in range(sampleSize)])
text = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "Corr":"ASM"].columns)] for i in range(sampleSize)])
shape = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "confidence":"eccentricity"].columns)] for i in range(sampleSize)])
bendingE = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "bendingE":"bendingE"].columns)] for i in range(sampleSize)])
other = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "z0":"z24"].columns)] for i in range(sampleSize)])

spec.reset_index(drop=True, inplace=True)
text.reset_index(drop=True, inplace=True)
shape.reset_index(drop=True, inplace=True)
bendingE.reset_index(drop=True, inplace=True)
other.reset_index(drop=True, inplace=True)
contrast.reset_index(drop=True, inplace=True)

spec_long = spec.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
text_long = text.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
shape_long = shape.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
bendingE_long = bendingE.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
other_long = other.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
contrast_long = contrast.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')

# Subset bending energy data to remove outliers
bendingE_Sub = pd.concat([data[i][dataOriginal[i]["bendingE"] < 100].loc[:, ["Orchard"] + list(data[i].loc[:, "bendingE":"bendingE"].columns)] for i in range(sampleSize)])
# bendingE_Sub = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "bendingE":"bendingE"].columns)] for i in range(sampleSize)])
bendingE_Sub.reset_index(drop=True, inplace=True)
bendingE_Sub_long = bendingE_Sub.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')

box_plot_comparison(spec_long, feature_group="Spectral")
box_plot_comparison(text_long, feature_group="Texture")
box_plot_comparison(shape_long, feature_group="Shape")
box_plot_comparison(bendingE_long, feature_group="Bending Energy")
box_plot_comparison(other_long, feature_group="Zernike")
box_plot_comparison(contrast_long, feature_group="Contrast")
box_plot_comparison(bendingE_Sub_long, feature_group="Bending Energy (Subset)")

# Variance barplots    
# Calculate variances for each feature in each orchard
variances = pd.DataFrame(columns=["Orchard", "Feature", "Variance"])
for i in range(sampleSize):
    orchard_data = data[i].loc[:, "confidence":]
    orchard_variances = orchard_data.var().reset_index()
    orchard_variances.columns = ["Feature", "Variance"]
    orchard_variances["Orchard"] = "Orchard {}".format(i+1)
    variances = pd.concat([variances, orchard_variances], ignore_index=True)

    spec = orchard_data.loc[:, "DSM":"OSAVI"]
    contrast = orchard_data.loc[:, "Contrast":"Contrast"]
    text = orchard_data.loc[:, "Corr":"ASM"]
    shape = orchard_data.loc[:, "roundness":"eccentricity"]
    zernicke = orchard_data.loc[:, "z0":"z24"]
    other = pd.DataFrame(orchard_data.loc[:, "confidence"])
    bend = pd.DataFrame(orchard_data.loc[:, "bendingE"])

variances_pivot = variances.pivot(index="Feature", columns="Orchard", values="Variance")
# Plot stacked barplot for each group
for group_name, group_data in [("Spectral", spec), ("Texture", text), ("Shape", shape), ("Other", other), ("Bending", bend), ("Zernicke", zernicke), ("Contrast", contrast)]:
    group_variances = group_data.columns
    group_variances_pivot = variances_pivot.loc[group_variances]
    group_variances_pivot.plot(kind='bar', stacked=False, figsize=(15, 8))
    plt.ylabel('Variance', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc='upper right', fontsize=25)
    plt.tight_layout()
    plt.savefig(f"results/EDA/Barplots/{group_name}.png")
    plt.show()


# %%
# Should also plot the Delauney Triangulation of the data
for i in range(sampleSize):
    d_w, d_g, d_p, d_c = tri.delauneyTriangulation(data[i])
    tri.delauneyPlot(d_g, d_p, d_c, img[i], f"results/EDA/Delauney/delauney_{i+1}.png")



# TODO: https://www.linkedin.com/pulse/quick-way-check-linearity-data-aditya-dutt/
#       Show linearity of data using PCA

for i in range(sampleSize):
    new_data = data[i].copy(deep=True)
    new_data.drop(columns=["Orchard", "Y"], inplace=True)
    new_data = new_data.loc[:, "confidence":]
    new_data = utils.engineer._scaleData(new_data)
    
    # Perform PCA
    pca = PCA()
    pca.fit(new_data)

    # Transform data
    pca_data = pca.transform(new_data)

    # Plot the first 10 eigenvalues
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 11), pca.explained_variance_[:10], marker='o', linestyle='--')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'First 10 Eigenvalues for Orchard {i+1}')
    plt.tight_layout()
    plt.savefig(f"results/EDA/PCA/eigenvalues_{i+1}.png")
    plt.show()


data_scaled = data[:]
for i in range(sampleSize):
    # data[i] = data[i].loc[:, ["Orchard"] + list(data[i].columns[:-1])]
    data_scaled[i].loc[:,'confidence':] = utils.engineer._scaleData(data_scaled[i].loc[:,'confidence':])

    # Group data by groups
spec = pd.concat([data_scaled[i].loc[:, ["Orchard"] + list(data_scaled[i].loc[:, "DSM":"OSAVI"].columns)] for i in range(sampleSize)])
text = pd.concat([data_scaled[i].loc[:, ["Orchard"] + list(data_scaled[i].loc[:, "Contrast":"ASM"].columns)] for i in range(sampleSize)])
shape = pd.concat([data_scaled[i].loc[:, ["Orchard"] + list(data_scaled[i].loc[:, "confidence":"bendingE"].columns)] for i in range(sampleSize)])

spec.reset_index(drop=True, inplace=True)
text.reset_index(drop=True, inplace=True)
shape.reset_index(drop=True, inplace=True)
other.reset_index(drop=True, inplace=True)

g = sns.PairGrid(shape, hue="Orchard", diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha=0.4)
g.map_diag(plt.hist, alpha=1, bins=30)
g.map_upper(sns.kdeplot, warn_singular=False)
g.add_legend(title='', fontsize=18)
for ax in g.axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
plt.gcf().subplots_adjust(left=0.1, bottom=0.1)  # Increase left border
plt.savefig("results/EDA/PairPlots/Shape.png")

g = sns.PairGrid(spec, hue="Orchard", diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha=0.4)
g.map_diag(plt.hist, alpha=1, bins=30)
g.map_upper(sns.kdeplot, warn_singular=False)
g.add_legend(title='', fontsize=18)
for ax in g.axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
plt.gcf().subplots_adjust(left=0.1, bottom=0.1)  # Increase left border
plt.savefig("results/EDA/PairPlots/Spec.png")

g = sns.PairGrid(text, hue="Orchard", diag_sharey=False, corner=False)
g.map_lower(plt.scatter, alpha=0.4)
g.map_diag(plt.hist, alpha=1, bins=30)
g.map_upper(sns.kdeplot, warn_singular=False)
g.add_legend(title='', fontsize=18)
for ax in g.axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
plt.gcf().subplots_adjust(left=0.1, bottom=0.1)  # Increase left border
plt.savefig("results/EDA/PairPlots/Text.png")

# %%
#     # calculate correlation values
#     # Recognise Multicollinearities
for i in range(sampleSize):
    orchard_spec = data_scaled[i].loc[:, "DSM":"OSAVI"]
    orchard_shape = data_scaled[i].loc[:, "confidence":"bendingE"]
    orchard_text = data_scaled[i].loc[:, "Contrast":"ASM"]

    heat = sns.clustermap(orchard_spec.corr(), 
                          annot=True, 
                          cbar_pos=None, 
                          cmap="plasma", 
                          annot_kws={"size": 18})
    ax_heatmap = heat.ax_heatmap  # For the heatmap itself
    ax_heatmap.tick_params(axis='both', which='major', labelsize=20)
    heat.ax_row_dendrogram.set_visible(False)
    heat.ax_col_dendrogram.set_visible(False)
    plt.gcf().subplots_adjust(left=-0.2, bottom=0.05, right=0.9, top=1.2)
    plt.savefig(f"results/EDA/ClusterMaps/spec_orchard_{i+1}.png")

    heat = sns.clustermap(orchard_shape.corr(), 
                          annot=True, 
                          cbar_pos=None,#(-0.1, .2, .03, .4), 
                          cmap="plasma", 
                          annot_kws={"size": 18})
    # plt.gcf().subplots_adjust(right=0.9, bottom=0.1)  # Increase right and bottom margins
    ax_heatmap = heat.ax_heatmap  # For the heatmap itself
    ax_heatmap.tick_params(axis='x', which='major', labelsize=20, rotation=90)
    ax_heatmap.tick_params(axis='y', which='major', labelsize=20, rotation=0)
    heat.ax_row_dendrogram.set_visible(False)
    heat.ax_col_dendrogram.set_visible(False)
    plt.gcf().subplots_adjust(left=-0.2, bottom=0.2, right=0.8, top=1.2)
    plt.savefig(f"results/EDA/ClusterMaps/shape_orchard_{i+1}.png")

    heat = sns.clustermap(orchard_text.corr(),
                          annot=True, 
                          cbar_pos=None, 
                          cmap="plasma", 
                          annot_kws={"size": 18})
    ax_heatmap = heat.ax_heatmap  # For the heatmap itself
    ax_heatmap.tick_params(axis='both', which='major', labelsize=20)
    heat.ax_row_dendrogram.set_visible(False)
    heat.ax_col_dendrogram.set_visible(False)
    plt.gcf().subplots_adjust(left=-0.2, bottom=0.05, right=0.9, top=1.2)
    plt.savefig(f"results/EDA/ClusterMaps/text_orchard_{i+1}.png")




# %%
# Demonstrate model performance before and after feature reduction
# Perform feature reduction

data_sensitive = list(dataOriginal[:])
fin_data = list(data_sensitive[0].loc[:,:'roundness'].columns) + ["compactness", "convexity", "solidity", "bendingE",  "DSM" , "NDRE", "OSAVI", "ASM", "Corr"]+ list(data_sensitive[0].loc[:,"z1":'z24'])
for i in range(sampleSize):
    data_sensitive[i] = data_sensitive[i].loc[:, fin_data]
    # data_sensitive[i].loc[:,'confidence':] = utils.engineer._scaleData(data_sensitive[i].loc[:, "confidence":])

# Perform anomaly detection
# Perform anomaly detection

# Obtain the precision for full dataset and precision for reduced dataset
# As well as images of improvements!
results = []
results_sensitive = []
for i in range(sampleSize):
    results.append(transductive.transductionResults(data[i], f"{i}"))
    results_sensitive.append(transductive.transductionResults(data_sensitive[i], f"{i}"))

# This is to separate the results

auroc, ap, _ = zip(*results)
auroc_sensitive, ap_sensitive, _ = zip(*results_sensitive)

auroc_df = pd.DataFrame()
ap_df = pd.DataFrame()

auroc_sensitive_df = pd.DataFrame()
ap_sensitive_df = pd.DataFrame()

for i in range(sampleSize):
    auroc_df = pd.concat([auroc_df, auroc[i]])
    ap_df = pd.concat([ap_df, ap[i]])

    auroc_sensitive_df = pd.concat([auroc_sensitive_df, auroc_sensitive[i]])
    ap_sensitive_df = pd.concat([ap_sensitive_df, ap_sensitive[i]])


# %%
