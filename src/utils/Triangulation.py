from libpysal import weights
from libpysal.cg import voronoi_frames
from libpysal import weights
import numpy as np
from contextily import add_basemap
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import shapely
import pandas as pd


def KNNGraph(data, nn = 3):

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
    knn = weights.KNN.from_dataframe(cases, k=nn)

    # Then, we can convert the graph to networkx object using the
    # .to_networkx() method.
    knn_graph = knn.to_networkx()

    # To plot with networkx, we need to merge the nodes back to
    # their positions in order to plot in networkx
    positions = dict(zip(knn_graph.nodes, coordinates))

    return knn, knn_graph, positions, cases

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

    return delaunay, delaunay_graph, positions, cells

# TODO: make sure the data being received is actually what we want.
#       should not subset data inside method
def setNodeAttributes(d_g, data):
    G = d_g
    # from confidence to distance 1 then from distance 4 till end
    # records = data.loc[:, "confidence":].to_dict('index')
    no_dists = list(data.columns)[4:18] + list(data.columns)[22:]
    records = data.loc[:, no_dists ].to_dict('index')

    # nodes now have attributes
    nx.set_node_attributes(G, records)
    return G

def distance(x, p1, p2, alpha = -1):
    """


    Notes:  Use the Inverse distance weighting
            because we want to give stronger weights to closer items
            Alpha <= 0    
    """
    position1 = x.iloc[p1]["centroid"]
    position2 = x.iloc[p2]["centroid"]

    return (shapely.distance(position1, position2) ** alpha)

def setEdgeAttributes(G, data):
    """
    
    """
    edges = [e for e in G.edges]

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

    return G


def delauneyPlot(d_g, d_p, v_cells, tryout,  plot_all = True):
    """
    Description:

    Attributes:

    Return:    
    """
    
    # Now, we can plot with a nice basemap.

    if (plot_all):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 15))

        v_cells.plot(ax=ax[0], 
                    facecolor="lightblue", 
                    alpha=0.50, 
                    edgecolor="cornsilk", 
                    linewidth=2)
        try:  # Try-except for issues with timeout/parsing failures in CI
            add_basemap(ax[0])
        except:
            pass

            ax[0].axis("off")
            ax[0].set_title("Delauney")
            nx.draw(d_g,
                    d_p,
                    ax=ax[0],
                    node_size=5,
                    node_color="k",
                    edge_color="k",
                    alpha=0.8)



            # plot 2
        tryout.plot.imshow(ax=ax[1])
        ax[1].axis("off")
        ax[1].set_title("Delauney")
        nx.draw(
        d_g,
        d_p,
        ax=ax[1],
        node_size=30,
        node_color="lightgreen",
        edge_color="red",
        alpha=0.8,
        )

            # plot 3
        nx.draw(d_g, ax=ax[2],node_size = 10, alpha = 0.8)
        ax[2].set_title("Delauney")
        plt.show()
    else:
        tryout.plot.imshow(ax=ax)
        ax.axis("off")
        ax.set_title("Delauney")
        nx.draw(
        d_g,
        d_p,
        ax=ax,
        node_size=30,
        node_color="lightgreen",
        edge_color="red",
        alpha=0.8,
        )
        plt.show()




def KNNPlot(knn_g, knn_p, cases, tryout, plot_all = True):
    """
    Description:

    Attributes:

    Return:    
    """
    cases.rename({"centroid": "geometry"}, axis="columns", inplace=True)
    if (plot_all):
            # plot with a nice basemap
        f, ax = plt.subplots(1, 3, figsize=(20, 20))
        # Plot 1
        cases.plot(marker=".", color="orangered", ax=ax[0])
        try:  # For issues with downloading/parsing basemaps in CI
            add_basemap(ax[0])
        except:
            pass
        ax[0].set_title("KNN-3")
        ax[0].axis("off")
        nx.draw(knn_g, knn_p, ax=ax[0], node_size=5, node_color="b")

        # Plot 2
        tryout.plot.imshow(ax=ax[1])
        ax[1].axis("off")
        ax[1].set_title("KNN-3")
        nx.draw(
            knn_g,
            knn_p,
            ax=ax[1],
            node_size=30,
            node_color="lightgreen",
            edge_color="red",
            alpha=0.8,
        )
        # Plot 3
        nx.draw(knn_g, ax=ax[2],node_size = 10, alpha = 0.8)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(25, 25))
        tryout.plot.imshow(ax=ax)
        ax.set_title("KNN-3")
        ax.axis("off")
        nx.draw(
            knn_g,
            knn_p,
            ax=ax,
            node_size=30,
            node_color="lightgreen",
            edge_color="red",
            alpha=0.8,
        )
        plt.show()