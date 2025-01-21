import matplotlib.pylab as plt
def plot(erf, normal, anomaly, name):
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
    fig, ax = plt.subplots(figsize=(20, 20))
    erf.plot.imshow(ax=ax)
    normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
    anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')
    fig.savefig(name)

def plotRef(erf, data, name):
    """
    Plots the given Earth Relief File (ERF) along with data regions colored by the 'Y' column.
    Parameters:
    erf : GeoDataFrame
        The Earth Relief File to be plotted as the base layer.
    data : GeoDataFrame
        The GeoDataFrame containing the data regions to be plotted and colored by the 'Y' column.
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')
    erf.plot.imshow(ax=ax)
    data.plot(column='Y', categorical=True, legend=True, ax=ax, cmap='rainbow', alpha=0.7)
    leg = ax.get_legend()
    for text in leg.get_texts():
        text.set_fontsize(20)  # Increase the font size of the legend
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.title("")  # Data Geometries Colored by Y
    fig.savefig(name)

def plotScores(erf, data, scores):
    
    _, ax = plt.subplots(1, figsize=(20, 20))
    erf.plot.imshow(ax=ax)
    data.assign(cl= scores).plot(column='cl', categorical=False,
            k=5, cmap='viridis', linewidth=0.1, ax=ax,
            edgecolor='white', legend=True, alpha=0.7)
    ax.set_axis_off()
    plt.title("Anomaly Scores")
    plt.show()