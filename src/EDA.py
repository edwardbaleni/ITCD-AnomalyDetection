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

from sklearn.feature_selection import VarianceThreshold

# plot boxplots for each group and separate by orchard
def box_plot_comparison(data, feature_group=None):
    # Filter by feature group if specified

    # Create box plots for each feature grouped by orchard
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Feature', y='Value', hue='Orchard', data=data)
    plt.xticks(rotation=90)
    # plt.title(f'Box Plot of Feature Values by Orchard ({feature_group or "All Groups"})')
    plt.ylabel('Value')
    plt.xlabel('Feature')
    plt.legend(loc='upper right', title='Orchard')
    plt.tight_layout()
    plt.savefig("results/EDA/Boxplots/box_plot_comparison_{}.png".format(feature_group))


def getDataNames(sampleSize):
    return utils.collectFiles(sampleSize)

def process(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped):
    # Your code goes here
    # num = 0
    # Do not scale data here because it will be split into training and testing data
    myData = utils.engineer(0,
                            [data_paths_tif], 
                            [data_paths_geojson], 
                            [data_paths_geojson_zipped],
                            False) # False)
    
    data = myData.data.copy(deep=True)
    mask = myData.mask
    spectralData = myData.spectralData

    # For plotting
    img = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
    img = img/255
    
    return data, img

if __name__ == '__main__':
    # Get sample size from user
    sampleSize = 5

    data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = getDataNames(sampleSize)

    # I have 20 cores!
    with Pool(5) as pool:
        args = zip(data_paths_tif, data_paths_geojson, data_paths_geojson_zipped)
        results = pool.starmap(process, list(args))

    data, img = zip(*results)
    data = list(data)
    
    # Plot all the images as they are
    # plot reference data
    for i in range(sampleSize):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.axis('off')
        img[i].plot.imshow(ax=ax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.title("")
        fig.savefig("results/EDA/Orchards/orchard_{}.png".format(i))
        plot.plotRef(img[i], data[i], "results/EDA/Orchards/reference_{}.png".format(i))


    # Label orchards according to orchard
    for i in range(sampleSize):
        data[i]["Orchard"] = "Orchard {}".format(i+1)
        data[i] = data[i].loc[:, ["Orchard"] + list(data[i].columns[:-1])]
    

    # Group data by groups
    spec = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "DEM_mean":"OSAVI_mean"].columns)] for i in range(sampleSize)])
    text = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "Contrast":"ASM"].columns)] for i in range(sampleSize)])
    shape = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "confidence":"bendingE"].columns)] for i in range(sampleSize)])
    other = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "z0":"z24"].columns)] for i in range(sampleSize)])
    

    spec.reset_index(drop=True, inplace=True)
    text.reset_index(drop=True, inplace=True)
    shape.reset_index(drop=True, inplace=True)
    other.reset_index(drop=True, inplace=True)

    # Scale features
    scaler = MinMaxScaler()

    spec.iloc[:, 1:] = scaler.fit_transform(spec.iloc[:, 1:])
    text.iloc[:, 1:] = scaler.fit_transform(text.iloc[:, 1:])
    shape.iloc[:, 1:] = scaler.fit_transform(shape.iloc[:, 1:])
    other.iloc[:, 1:] = scaler.fit_transform(other.iloc[:, 1:])


    spec_long = spec.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
    text_long = text.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
    shape_long = shape.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')
    other_long = other.melt(id_vars=['Orchard'], var_name='Feature', value_name='Value')

    # Example Usage
    box_plot_comparison(spec_long, feature_group="Texture")
    box_plot_comparison(text_long, feature_group="Spectral")
    box_plot_comparison(shape_long, feature_group="Shape")
    box_plot_comparison(other_long, feature_group="Zernike")


    # Obtain number of outliers each orchard and number of delineations
    OutlierInfo = pd.DataFrame(columns=["Orchard", "Outliers", "Delineations", "Ratio"])
    for i in range(sampleSize):
        new_row = {"Orchard": "Orchard {}".format(i+1), 
                   "Outliers": data[i].loc[data[i]["Y"] == "Outlier"].shape[0], 
                   "Delineations": data[i].shape[0]}
        new_row["Ratio"] = new_row["Outliers"]/new_row["Delineations"]
        OutlierInfo = pd.concat([OutlierInfo, pd.DataFrame([new_row])], ignore_index=True)

    # Export benchmark data as a CSV
    # TODO: Need to do for 30 samples
    OutlierInfo.to_csv("results/EDA/benchmark_data.csv", index=False)



    # # %%

    #     # TODO: For the spectral indices and vegetative indices we
    #     #       just need to do profile plots to understand relevance.

    #     # TODO: What we can do for the EDA is look at morphological and image properties separately
    #     #       Then look at the most significant of these together


    # # %% 

        # Group data by groups
    spec = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "DEM_mean":"OSAVI_mean"].columns)] for i in range(sampleSize)])
    text = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "Contrast":"ASM"].columns)] for i in range(sampleSize)])
    shape = pd.concat([data[i].loc[:, ["Orchard"] + list(data[i].loc[:, "confidence":"bendingE"].columns)] for i in range(sampleSize)])

    spec.reset_index(drop=True, inplace=True)
    text.reset_index(drop=True, inplace=True)
    shape.reset_index(drop=True, inplace=True)
    other.reset_index(drop=True, inplace=True)

    # # Scale features
    # scaler = MinMaxScaler()

    # spec.iloc[:, 1:] = scaler.fit_transform(spec.iloc[:, 1:])
    # text.iloc[:, 1:] = scaler.fit_transform(text.iloc[:, 1:])
    # shape.iloc[:, 1:] = scaler.fit_transform(shape.iloc[:, 1:])







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

    # # TODO: log DEM_mean after demonstrating that it should be logged here!
    # #       however, does transforming the data stutter detection or improve it?
    g = sns.PairGrid(shape, hue="Orchard", diag_sharey=False, corner=False)
    g.map_lower(plt.scatter, alpha=0.4)
    g.map_diag(plt.hist, alpha=1, bins=30)
    g.map_upper(sns.kdeplot, warn_singular=False)
    g.add_legend()
    plt.savefig("results/EDA/PairPlots/pairplot_shape.png")

    g = sns.PairGrid(spec, hue="Orchard", diag_sharey=False, corner=False)
    g.map_lower(plt.scatter, alpha=0.4)
    g.map_diag(plt.hist, alpha=1, bins=30)
    g.map_upper(sns.kdeplot, warn_singular=False)
    g.add_legend()
    plt.savefig("results/EDA/PairPlots/pairplot_spec.png")

    g = sns.PairGrid(text, hue="Orchard", diag_sharey=False, corner=False)
    g.map_lower(plt.scatter, alpha=0.4)
    g.map_diag(plt.hist, alpha=1, bins=30)
    g.map_upper(sns.kdeplot, warn_singular=False)
    g.add_legend()
    plt.savefig("results/EDA/PairPlots/pairplot_text.png")

    #     # calculate correlation values
    #     # Recognise Multicollinearities
    for i in range(sampleSize):
        orchard_spec = data[i].loc[:, "DEM_mean":"OSAVI_mean"]
        orchard_shape = data[i].loc[:, "confidence":"bendingE"]
        orchard_text = data[i].loc[:, "Contrast":"ASM"]

        sns.clustermap(orchard_spec.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap="plasma")
        plt.savefig(f"results/EDA/ClusterMaps/clustermap_spec_orchard_{i+1}.png")
        sns.clustermap(orchard_shape.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap="plasma")
        plt.savefig(f"results/EDA/ClusterMaps/clustermap_shape_orchard_{i+1}.png")
        sns.clustermap(orchard_text.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap="plasma")
        plt.savefig(f"results/EDA/ClusterMaps/clustermap_text_orchard_{i+1}.png")

    
    # Other is not necessary as we already know that Zernicke moments are independent
    # however, we need to compare the confidence and DEM_mean or keep these as individual features


    # # %%    
    #                     # Feature Selection

    # It is necessary to get an idea of the variances of the features in the 
    # scaling that will be used for the rest of the paper.
    # In this paper, RobustScaling will be used. So first we will scale the data
    # Then separate the data into the different groups
    # Then obtain barplots of the variances of the features in each group over each orchard
    # Then this will help inform how to use variance thresholding to remove low variance features

    # TODO: Using a conservative threshold of 0.5 we can remove z0

    # Scale features
    data_scaled = data[:]
    for i in range(sampleSize):
        data_scaled[i].loc[:,'confidence':] = utils.engineer._scaleData(data_scaled[i].loc[:,'confidence':])

    # Calculate variances for each feature in each orchard
    variances = pd.DataFrame(columns=["Orchard", "Feature", "Variance"])
    for i in range(sampleSize):
        orchard_data = data_scaled[i].loc[:, "confidence":]
        orchard_variances = orchard_data.var().reset_index()
        orchard_variances.columns = ["Feature", "Variance"]
        orchard_variances["Orchard"] = "Orchard {}".format(i+1)
        variances = pd.concat([variances, orchard_variances], ignore_index=True)

        spec = data_scaled[i].loc[:, "DEM_mean":"OSAVI_mean"]
        text = data_scaled[i].loc[:, "Contrast":"ASM"]
        shape = data_scaled[i].loc[:, "roundness":"eccentricity"]
        zernicke = data_scaled[i].loc[:, "z0":"z24"]
        other = pd.DataFrame(data_scaled[i].loc[:, "confidence"])
        bend = pd.DataFrame(data_scaled[i].loc[:, "bendingE"])

    # Pivot the data for plotting
    variances_pivot = variances.pivot(index="Feature", columns="Orchard", values="Variance")
    # Plot stacked barplot for each group
    for group_name, group_data in [("Spectral", spec), ("Texture", text), ("Shape", shape), ("Other", other), ("Bending", bend), ("Zernicke", zernicke)]:
        group_variances = group_data.columns
        group_variances_pivot = variances_pivot.loc[group_variances]
        
        group_variances_pivot.plot(kind='bar', stacked=False, figsize=(15, 8))
        plt.ylabel('Variance')
        # plt.title(f'Grouped Barplot of Feature Variances Across Orchards ({group_name} Group)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"results/EDA/Barplots/grouped_barplot_variances_{group_name.lower()}.png")
        plt.show()



    X = data_scaled[0].loc[:,"confidence":]

    selector = VarianceThreshold(threshold=0.5)

    selector.fit_transform(X)

    # outputting low variance columns
    concol = [column for column in data_scaled[0].loc[:,"confidence":].columns 
            if column not in data_scaled[0].loc[:,"confidence":].columns[selector.get_support()]]

    for features in concol:
        print(features)

    # # drop low variance columns
    # X.drop(concol, axis = 1)

    # # TODO: Do pairplot and clustermap for all remaining features after
    # #       feature selection
    # #       Only after feature selection
    
    # # Plot clustermap for all orchards using data_scaled
    # for i in range(sampleSize):
    #     orchard_data = data_scaled[i].loc[:, "confidence":]
    #     sns.clustermap(orchard_data.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap="plasma")
    #     plt.savefig(f"results/EDA/clustermap_all_features_orchard_{i+1}.png")
