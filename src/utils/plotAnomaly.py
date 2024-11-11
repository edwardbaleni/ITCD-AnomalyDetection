import matplotlib.pylab as plt
def plot(erf, normal, anomaly):
    """
    Plots the given Earth Relief File (ERF) along with normal and anomaly regions.
    Parameters:
    erf : GeoDataFrame
        The Earth Relief File to be plotted as the base layer.
    normal : GeoDataFrame
        The GeoDataFrame containing the normal regions to be plotted with red edges.
    anomaly : GeoDataFrame
        The GeoDataFrame containing the anomaly regions to be plotted with blue edges.
    Returns:
    None
    """

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    erf.plot.imshow(ax=ax)
    normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
    anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')