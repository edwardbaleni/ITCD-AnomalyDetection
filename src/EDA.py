import utils

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import shapely

import seaborn as sns
import plotly.express as px
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from numpy import mean, median, std
from scipy.stats import kurtosis, skew

import utils.plotAnomaly as plot

# plot boxplots for each group and separate by orchard
def box_plot_comparison(data, feature_group=None):
    # Filter by feature group if specified
    if feature_group:
        data = data[data['Group'] == feature_group]

    # Create box plots for each feature grouped by orchard
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Feature', y='Value', hue='Orchard', data=data)
    plt.xticks(rotation=90)
    plt.title(f'Box Plot of Feature Values by Orchard ({feature_group or "All Groups"})')
    plt.ylabel('Value')
    plt.xlabel('Feature')
    plt.legend(loc='upper right', title='Orchard')
    plt.tight_layout()
    plt.savefig(f"results/EDA/box_plot_comparison_{feature_group or 'all'}.png")

# # plot boxplots for each group and separate by orchard
# def box_plot_comparison(data, feature_group=None):
#     # Filter by feature group if specified
#     if feature_group:
#         data = data[data['Group'] == feature_group]

#     # Create box plots for each feature grouped by orchard
#     features = data['Feature'].unique()
#     for feature in features:
#         plt.figure(figsize=(15, 8))
#         sns.boxplot(x='Orchard', y='Value', data=data[data['Feature'] == feature])
#         plt.xticks(rotation=90)
#         plt.title(f'Box Plot of {feature} Values by Orchard ({feature_group or "All Groups"})')
#         plt.ylabel('Value')
#         plt.xlabel('Orchard')
#         plt.legend(loc='upper right', title='Orchard')
#         plt.tight_layout()
#         plt.savefig(f"results/EDA/box_plot_comparison_{feature_group or 'all'}_{feature}.png")
#         plt.close()


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
    
    # Plot all the images as they are
    # plot reference data
    for i in range(sampleSize):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.axis('off')
        img[i].plot.imshow(ax=ax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.title("")
        fig.savefig("results/EDA/orchard_{}.png".format(i))
        plot.plotRef(img[i], data[i], "results/EDA/reference_{}.png".format(i))


    # Label orchards according to orchard
    for i in range(sampleSize):
        data[i]["orchard"] = "Orchard {}".format(i)
    

    # Group data by groups
    spec = pd.concat([data[i].loc[:, ["orchard"] + list(data[i].loc[:, "NIR_mean":"OSAVI_mean"].columns)] for i in range(sampleSize)])
    text = pd.concat([data[i].loc[:, ["orchard"] + list(data[i].loc[:, "Contrast":"ASM"].columns)] for i in range(sampleSize)])
    shape = pd.concat([data[i].loc[:, ["orchard"] + list(data[i].loc[:, "roundness":"bendingE"].columns)] for i in range(sampleSize)])

    spec.reset_index(drop=True, inplace=True)
    text.reset_index(drop=True, inplace=True)
    shape.reset_index(drop=True, inplace=True)

    # Scale features
    scaler = MinMaxScaler()

    spec.iloc[:, 1:] = scaler.fit_transform(spec.iloc[:, 1:])
    text.iloc[:, 1:] = scaler.fit_transform(text.iloc[:, 1:])
    shape.iloc[:, 1:] = scaler.fit_transform(shape.iloc[:, 1:])

    # Prepare data for box plot comparison
    combined_data = pd.concat([
        pd.melt(spec, id_vars=['orchard'], var_name='Feature', value_name='Value').assign(Group='Spectral'),
        pd.melt(text, id_vars=['orchard'], var_name='Feature', value_name='Value').assign(Group='Texture'),
        pd.melt(shape, id_vars=['orchard'], var_name='Feature', value_name='Value').assign(Group='Shape')
    ])

    combined_data.rename(columns={'orchard': 'Orchard'}, inplace=True)

    # Example Usage
    box_plot_comparison(combined_data, feature_group="Texture")
    box_plot_comparison(combined_data, feature_group="Spectral")
    box_plot_comparison(combined_data, feature_group="Shape")
    box_plot_comparison(combined_data)


    # Create a pivot table for the data
    # This is what the Radar Chart will be based off
    descr = combined_data.pivot_table("Value", index=["Orchard", "Feature"], aggfunc=[mean, median, std, kurtosis, skew])

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from plotly.express import parallel_coordinates

# # Sample Data Creation
# def create_sample_data():
#     np.random.seed(42)  # For reproducibility
#     data = {
#         'Image': [f'Image_{i}' for i in range(1, 6) for _ in range(54)],
#         'Feature': [f'Feature_{i}' for _ in range(5) for i in range(1, 55)],
#         'Value': np.random.rand(54 * 5),
#         'Group': [
#             "Texture" if i <= 18 else "Spectral" if i <= 36 else "Shape"
#             for _ in range(5) for i in range(1, 55)
#         ]
#     }
#     return pd.DataFrame(data)

# # Generate Data
# created_data = create_sample_data()

# # Parallel Coordinates Plot
# def parallel_coordinates_plot(data, group):
#         # Filter data for the selected group
#         group_data = data[data['Group'] == group]

#         # Reshape data to wide format for parallel coordinates
#         pivot_data = group_data.pivot_table(
#                 index='Image', columns='Feature', values='Value'
#         ).reset_index()

#         # Use plotly for an interactive parallel plot
#         fig = parallel_coordinates(
#                 pivot_data,
#                 color=pivot_data.columns[1],
#                 title=f"Parallel Coordinates Plot of {group} Features by Image",
#                 labels={col: col for col in pivot_data.columns},
#                 dimensions=pivot_data.columns[1:],
#         )
#         fig.show()


# # Radar Chart Plot
# def radar_chart_plot(data, group):
#     # Filter data for the selected group
#     group_data = data[data['Group'] == group]

#     # Aggregate values by image and feature
#     radar_data = group_data.groupby(['Image', 'Feature'])['Value'].mean().unstack()

#     # Normalize data for better comparison
#     radar_data = radar_data.div(radar_data.max(axis=1), axis=0)

#     # Prepare data for radar chart
#     categories = radar_data.columns.tolist()
#     num_vars = len(categories)

#     # Plot Radar Chart
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

#     for i, (image, row) in enumerate(radar_data.iterrows()):
#         values = row.tolist()
#         values += values[:1]  # Close the radar chart
#         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#         angles += angles[:1]

#         ax.plot(angles, values, label=image)
#         ax.fill(angles, values, alpha=0.25)

#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)

#     # Draw feature labels
#     ax.set_xticks(np.linspace(0, 2 * np.pi, num_vars, endpoint=False))
#     ax.set_xticklabels(categories)

#     # Draw y-labels
#     ax.set_yticks([0.25, 0.5, 0.75, 1.0])
#     ax.set_yticklabels(["25%", "50%", "75%", "100%"])

#     ax.set_title(f"Radar Chart for {group} Features", size=20, pad=20)
#     ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

#     plt.show()

# # # Run Visualizations
# # parallel_coordinates_plot(created_data, group="Texture")
# radar_chart_plot(created_data, group="Texture")


# import seaborn as sns
# import matplotlib.pyplot as plt

# def box_plot_comparison(data, feature_group=None):
#     # Filter by feature group if specified
#     if feature_group:
#         data = data[data['Group'] == feature_group]

#     # Create box plots for each feature grouped by image
#     plt.figure(figsize=(15, 8))
#     sns.boxplot(x='Feature', y='Value', hue='Image', data=data)
#     plt.xticks(rotation=90)
#     plt.title(f'Box Plot of Feature Values by Image ({feature_group or "All Groups"})')
#     plt.ylabel('Value')
#     plt.xlabel('Feature')
#     plt.legend(loc='upper right', title='Image')
#     plt.tight_layout()
#     plt.show()

# # Example Usage
# box_plot_comparison(data, feature_group="Texture")


    # # %%

    #     # TODO: For the spectral indices and vegetative indices we
    #     #       just need to do profile plots to understand relevance.

    #     # TODO: What we can do for the EDA is look at morphological and image properties separately
    #     #       Then look at the most significant of these together


    # # %% 

    # shape = data.loc[:, "roundness":"bendingE"]
    # # dist = data.loc[:, "dist1":"dist4"]
    # spec = data.loc[:, "DEM_mean":"OSAVI_mean"]
    # # we already know that Zernke polynomials are independent
    # tex = data.loc[:, "Contrast":]

    # from pypalettes import get_hex
    # palette = get_hex("VanGogh3", keep_first_n=8)


    # # %%
    # # Note that the palette is set as a global variable can change this later.
    # def boxplot(dat, lo = True):
    #     sns.set_theme(style="darkgrid")
    #     Props = {'boxprops':{"alpha":0.7, 'edgecolor':palette[3], 'facecolor':palette[2]},
    #             'medianprops':{'color':palette[3]},
    #             'whiskerprops':{'color':palette[3]},
    #             'capprops':{'color':palette[3]},
    #             'flierprops':{'color':palette[3]}
    #             }

    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     if (lo):
    #         ax.set(yscale="log")
    #     sns.boxplot(data=dat, 
    #                 ax=ax, 
    #                 linewidth=2,
    #                 #color=palette[0], 
    #                 **Props)

    #     ax.tick_params("x", labelrotation=45)

    # boxplot(spec)
    # boxplot(shape.iloc[:,:-1])
    # boxplot(shape[["bendingE"]])
    # # boxplot(dist, False)

    # # %%

    # #
    # # TODO: log DEM_mean after demonstrating that it should be logged here!
    # #       however, does transforming the data stutter detection or improve it?
    # g = sns.PairGrid(shape, diag_sharey=False, corner=False)
    # g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
    # g.map_diag(plt.hist, alpha = 1, bins=30, color = palette[3])
    # g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

    # # g = sns.PairGrid(dist, diag_sharey=False, corner=False)
    # # g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
    # # g.map_diag(plt.hist, alpha = 1, bins=30,color = palette[3])
    # # g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

    # g = sns.PairGrid(spec, diag_sharey=False, corner=False)
    # g.map_lower(plt.scatter, alpha = 0.4, color=palette[2])
    # g.map_diag(plt.hist, alpha = 1, bins=30,color = palette[3])
    # g.map_upper(sns.kdeplot, color=palette[2], warn_singular=False)

    # # %%    
    #                     # Feature Selection

    # # %%
    #     # calculate correlation values
    #     # Recognise Multicollinearities
    # sns.clustermap(spec.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")# palette)
    # # sns.clustermap(dist.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")
    # sns.clustermap(shape.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")
    # sns.clustermap(tex.corr(), annot=True, cbar_pos=(-0.1, .2, .03, .4), cmap = "plasma")# palette)

    # # %%

    # from statsmodels.stats.outliers_influence import variance_inflation_factor

    # # the independent variables set
    # X = data.loc[:,"confidence":]

    # # VIF dataframe
    # vif_data = pd.DataFrame()
    # vif_data["feature"] = X.columns

    # # calculating VIF for each feature
    # vif_data["VIF"] = [variance_inflation_factor(X.values, i)
    #                         for i in range(len(X.columns))]

    # print(vif_data)

    # # %%

    # from sklearn.feature_selection import VarianceThreshold
    # from sklearn.decomposition import PCA

    # X = data.loc[:,"confidence":]

    # selector = VarianceThreshold(threshold=1)

    # selector.fit_transform(X)

    # # outputting low variance columns
    # concol = [column for column in data.loc[:,"confidence":].columns 
    #         if column not in data.loc[:,"confidence":].columns[selector.get_support()]]

    # for features in concol:
    #     print(features)

    # # drop low variance columns
    # X.drop(concol, axis = 1)

    # # %%

    # # Can use PCA to exhibit linearity
    # # conversation should be in terms of eigenvalues not principle components
    # from sklearn.decomposition import PCA
    # # Perform PCA
    # pca = PCA()
    # pca.fit(X)

    # # Get eigenvalues
    # eigenvalues = pca.explained_variance_

    # # Plot eigenvalues
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    # plt.title('Eigenvalues of the Features')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    # plt.grid(True)
    # plt.show()
