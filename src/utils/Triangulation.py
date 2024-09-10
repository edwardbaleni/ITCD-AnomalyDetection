from libpysal import weights
from libpysal.cg import voronoi_frames
from libpysal import weights
import numpy as np


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

    return knn, knn_graph, positions

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

    return delaunay, delaunay_graph, positions
