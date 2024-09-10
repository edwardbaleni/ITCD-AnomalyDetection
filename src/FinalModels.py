# %%
import dataHandler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.plotting
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator

sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = dataHandler.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = dataHandler.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
                              scale=True)
data = myData.data
delineations = myData.delineations
mask = myData.mask
spectralData = myData.spectralData
# For plotting
mask = mask
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255


# remove non-robust features - Doesn't help yet
# data = data[list(data.columns)[:5] + list(data.columns)[10:]]
# %%

                    # Extended Isolation Forest

# %%
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html#examples
# https://github.com/sahandha/eif/blob/master/Notebooks/TreeVisualization.ipynb 
# Set the predictors
h2o.init()
h2o_df = h2o.H2OFrame(data[list(data.columns)[4:]])
predictors = list(data.columns)[4:]
# https://github.com/sahandha/eif/blob/master/Notebooks/EIF.ipynb
    # Maybe this may help with plotting but I am uncertain
# %%
# Extended Isolation Forest is a great unsupervised method for anomaly detection
# however, it does not allow for the use of spatial features

# Define an Extended Isolation forest model
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 1000,
                                          sample_size = int(data.shape[0] * 0.8),
                                          extension_level = 6)#len(predictors) - 1)

# Train Extended Isolation Forest
eif.train(x = predictors,
          training_frame = h2o_df)

# Calculate score
eif_result = eif.predict(h2o_df)

# Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
# or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
anomaly_score = eif_result["anomaly_score"]

# Average path length  of the point in Isolation Trees from root to the leaf
mean_length = eif_result["mean_length"]

# %%
b = eif_result.as_data_frame()

    # 0.5 is a good threshold, for a weak one go <= 0.4
    # for a tight one go >= 0.5 
anomaly = data[b["anomaly_score"] > 0.4]
nominal = data[b["anomaly_score"] <= 0.4]

# %%


# Plotting
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
nominal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')

# %%



# %%
                    # Delauney Paper
# %%

# https://gis.stackexchange.com/questions/459091/definition-of-multipolygon-distance-in-shapely
import shapely.plotting

# shapely.plotting.plot_polygon(data.iloc[0,1], color = "red")
# shapely.plotting.plot_polygon(data.iloc[4,1], color = "blue")
# plt.show()

# # this distance outputs the distance from the nearest vertex to the nearest vertex of the
# # polygons not from the centroid to the centroid
# print("distance: ", {data.iloc[0,1].distance(data.iloc[4,1])})

# (-8.28502 - -8.28497) = 0.00005       # From closrset vertex to closest vertex
# (-8.28506 - -8.28495) = 0.00011       # From centroid to centroid\

# don't have to do polygons to polygons can do centroid to centtroid.

# %% 
                    # Delauney Paper


# %%


def KNNGraph(data, nn = 3):
    from libpysal import weights, examples
    from contextily import add_basemap
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import geopandas

    # read in example data from a geopackage file. Geopackages
    # are a format for storing geographic data that is backed
    # by sqlite. geopandas reads data relying on the fiona package,
    # providing a high-level pandas-style interface to geographic data.
    cases = data[["centroid"]]
    cases.rename({"centroid": "geometry"}, axis="columns", inplace=True)

    # construct the array of coordinates for the centroid
    coordinates = np.column_stack((cases.geometry.x, cases.geometry.y))

    # construct two different kinds of graphs:

    ## 3-nearest neighbor graph, meaning that points are connected
    ## to the three closest other points. This means every point
    ## will have exactly three neighbors.
    knn3 = weights.KNN.from_dataframe(cases, k=nn)

    ## The 50-meter distance band graph will connect all pairs of points
    ## that are within 50 meters from one another. This means that points
    ## may have different numbers of neighbors.
    dist = weights.DistanceBand.from_array(coordinates, threshold=50)

    # Then, we can convert the graph to networkx object using the
    # .to_networkx() method.
    knn_graph = knn3.to_networkx()
    dist_graph = dist.to_networkx()

    # To plot with networkx, we need to merge the nodes back to
    # their positions in order to plot in networkx
    positions = dict(zip(knn_graph.nodes, coordinates))

    # plot with a nice basemap
    f, ax = plt.subplots(1, 2, figsize=(20, 20))
    for i, facet in enumerate(ax):
        cases.plot(marker=".", color="orangered", ax=facet)
        try:  # For issues with downloading/parsing basemaps in CI
            add_basemap(facet)
        except:
            pass
        facet.set_title(("KNN-3", "50-meter Distance Band")[i])
        facet.axis("off")
    nx.draw(knn_graph, positions, ax=ax[0], node_size=5, node_color="b")
    nx.draw(dist_graph, positions, ax=ax[1], node_size=5, node_color="b")
    plt.show()

    return knn_graph, positions



# %%
from libpysal import weights, examples
from libpysal.cg import voronoi_frames
from contextily import add_basemap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
def delauneyTriangulation(data):

    # read in example data from a geopackage file. Geopackages
    # are a format for storing geographic data that is backed
    # by sqlite. geopandas reads data relying on the fiona package,
    # providing a high-level pandas-style interface to geographic data.
    # Many different kinds of geographic data formats can be read by geopandas.
    cases = data["centroid"]

    # In order for networkx to plot the nodes of our graph correctly, we
    # need to construct the array of coordinates for each point in our dataset.
    # To get this as a numpy array, we extract the x and y coordinates from the
    # geometry column.
    coordinates = np.column_stack((cases.geometry.x, cases.geometry.y))

    # While we could simply present the Delaunay graph directly, it is useful to
    # visualize the Delaunay graph alongside the Voronoi diagram. This is because
    # the two are intrinsically linked: the adjacency graph of the Voronoi diagram
    # is the Delaunay graph for the set of generator points! Put simply, this means
    # we can build the Voronoi diagram (relying on scipy.spatial for the underlying
    # computations), and then convert these polygons quickly into the Delaunay graph.
    # Be careful, though; our algorithm, by default, will clip the voronoi diagram to
    # the bounding box of the point pattern. This is controlled by the "clip" argument.
    cells, generators = voronoi_frames(coordinates, clip="convex hull")

    # With the voronoi polygons, we can construct the adjacency graph between them using
    # "Rook" contiguity. This represents voronoi cells as being adjacent if they share
    # an edge/face. This is an analogue to the "von Neuman" neighborhood, or the 4 cardinal
    # neighbors in a regular grid. The name comes from the directions a Rook piece can move
    # on a chessboard.
    delaunay = weights.Rook.from_dataframe(cells)

    # Once the graph is built, we can convert the graphs to networkx objects using the
    # relevant method.
    delaunay_graph = delaunay.to_networkx()

    # To plot with networkx, we need to merge the nodes back to
    # their positions in order to plot in networkx
    positions = dict(zip(delaunay_graph.nodes, coordinates))
    
    # Now, we can plot with a nice basemap.

    ax = cells.plot(figsize = (25,25),
                    facecolor="lightblue", 
                    alpha=0.50, 
                    edgecolor="cornsilk", 
                    linewidth=2)
    try:  # Try-except for issues with timeout/parsing failures in CI
        add_basemap(ax)
    except:
        pass

    ax.axis("off")
    nx.draw(
        delaunay_graph,
        positions,
        ax=ax,
        node_size=5,
        node_color="k",
        edge_color="k",
        alpha=0.8,
    )
    plt.show()

    return delaunay_graph, positions


delaunay_graph, positions = delauneyTriangulation(data)

# %%

delaunay_graph, positions = delauneyTriangulation(data)

fig, ax = plt.subplots(figsize=(25, 25))
tryout.plot.imshow(ax=ax)
ax.axis("off")
nx.draw(
    delaunay_graph,
    positions,
    ax=ax,
    node_size=30,
    node_color="lightgreen",
    edge_color="red",
    alpha=0.8,
)
plt.show()

knn_graph, positions2 = KNNGraph(data)


fig, ax = plt.subplots(figsize=(25, 25))
tryout.plot.imshow(ax=ax)
ax.axis("off")
nx.draw(
    knn_graph,
    positions,
    ax=ax,
    node_size=30,
    node_color="lightgreen",
    edge_color="red",
    alpha=0.8,
)
plt.show()

# %%
# check if any nodes have no neighbours
# delaunay.islands

# delaunay.neighbors

delaunay_graph.edges

# %%
G = delaunay_graph
# from confidence to distance 1 then from distance 4 till end
# records = data.loc[:, "confidence":].to_dict('index')
no_dists = list(data.columns)[4:18] + list(data.columns)[22:]
records = data.loc[:, no_dists ].to_dict('index')

# nodes now have attributes
nx.set_node_attributes(G, records)
G.nodes[1522]

# %%
edges = [e for e in delaunay_graph.edges]

# Use the Inverse distance weighting
# because we want to give stronger weights to closer items
# Alpha <= 0
def distance(x, p1, p2, alpha = -1):
    position1 = x.iloc[p1]["centroid"]
    position2 = x.iloc[p2]["centroid"]

    return (shapely.distance(position1, position2) ** alpha)

# These numbers are not scaled, but edges only have one
# attribute so I don't think it is necessary to scale them
attribute_dict = {}
while edges != []:
    e = edges[0]
    # # Can't scale it with this commented out method
    # if G.edges[e]['weight'] == 1.0:
    #     G.edges[e]['weight'] = distance(data, e[0], e[1])
    #     edges.pop(0)
    attribute_dict[e] = {"distance" : distance(data, e[0], e[1])}
    edges.pop(0)

# now we can scale distances
distances = pd.DataFrame.from_dict(attribute_dict, "index")
distances = (distances - distances.mean()) / distances.std()
attribute_dict = distances.to_dict("index")
# Add attributes to network
nx.set_edge_attributes(G, attribute_dict)




# %%
# TODO: https://stackoverflow.com/questions/70452465/how-to-load-in-graph-from-networkx-into-pytorch-geometric-and-set-node-features
#       https://stackoverflow.com/questions/71011514/converting-a-pyg-graph-to-a-networkx-graph
import torch
from torch_geometric.utils.convert import from_networkx

pyg_graph = from_networkx(G, 
                          group_node_attrs = "all", 
                          group_edge_attrs= "distances")

# %%
# TODO: AD with just graph structure
# TODO: AD with graph structure plus attributes
    # TODO: https://docs.pygod.org/en/latest/tutorials/1_intro.html#sphx-glr-tutorials-1-intro-py
    # TODO: https://pytorch-geometric.readthedocs.io/en/latest/index.html

# train a dominant detector
from pygod.detector import CoLA

model = CoLA(gpu=0)  # hyperparameters can be set here # gpu = 0, uses gpu # gpu = 1 uses cpu
model.fit(pyg_graph)  # input data is a PyG data object

# get outlier scores on the training data (transductive setting)
label = model.label_
labels = label.detach().cpu().numpy()
score = model.decision_score_

# %%
anomaly = data[labels == 1]
nominal = data[labels == 0]

# Plotting
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
nominal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')

# %%
# train a dominant detector
from pygod.detector import DMGD

model = DMGD()  # hyperparameters can be set here # gpu = 0, uses gpu # gpu = 1 uses cpu
model.fit(pyg_graph)  # input data is a PyG data object

# get outlier scores on the training data (transductive setting)
label = model.label_
labels = label.detach().cpu().numpy()
score = model.decision_score_

anomaly = data[labels == 1]
nominal = data[labels == 0]

# Plotting
fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
nominal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')


# %%
# TODO: performance metrics for outlier detection
#        https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#sphx-glr-auto-examples-miscellaneous-plot-outlier-detection-bench-py



# %%
# For Correction purposes
    # False-positives
        # Remove observation
    # Over-segmentations   
        # TODO: https://medium.com/@jesse419419/understanding-iou-and-nms-by-a-j-dcebaad60652
        #       https://infoscience.epfl.ch/server/api/core/bitstreams/768fc3bc-f7d4-4533-825d-5c398995526d/content
    # Under-segmentations
        # https://arxiv.org/pdf/2202.08682
        # https://www.mdpi.com/2072-4292/12/5/767
    # False-negatives - https://www.mdpi.com/2072-4292/11/4/410
        # YOlOv10
        # Can train YOLO on all available orchards
        # following this we can use it as a trianed model.
        # we then isolate the area that has a potential false-negative and test using trained YOLOv10.

    # Can build probability map using - https://www.mdpi.com/2072-4292/12/5/767