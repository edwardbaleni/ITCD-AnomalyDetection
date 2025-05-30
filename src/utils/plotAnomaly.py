from matplotlib.lines import Line2D

import matplotlib.pylab as plt
def plot(img, normal, anomaly, name):
    """
    Plots the given Earth Relief File (ERF) along with normal and anomaly regions.
    Parameters:
    img : GeoDataFrame
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
    img.plot.imshow(ax=ax)
    ax.axis('off')
    normal.plot(ax=ax, facecolor = 'none', edgecolor='red', label='Normal Regions') 
    anomaly.plot(ax=ax, facecolor = 'none', edgecolor='blue', label='Anomaly Regions')
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2)]
    ax.legend(custom_lines, ['Normal', 'Outliers'], loc='upper right', fontsize=25)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.title("")  # Data Geometries Colored by Y
    fig.savefig(name)

def plotRef(img, data, name):
    """
    Plots the given Earth Relief File (ERF) along with data regions colored by the 'Y' column.
    Parameters:
    img : GeoDataFrame
        The Earth Relief File to be plotted as the base layer.
    data : GeoDataFrame
        The GeoDataFrame containing the data regions to be plotted and colored by the 'Y' column.
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')
    img.plot.imshow(ax=ax)
    data.plot(column='Y', categorical=True, legend=True, ax=ax, cmap='rainbow', alpha=0.7)
    leg = ax.get_legend()
    for text in leg.get_texts():
        text.set_fontsize(20)  # Increase the font size of the legend
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.title("")  # Data Geometries Colored by Y
    fig.savefig(name)

def plotScores(erf, data, scores, name):
    
    _, ax = plt.subplots(1, figsize=(20, 20))
    erf.plot.imshow(ax=ax)
    ax.axis('off')
    data.assign(cl= scores).plot(column='cl', categorical=False,
            k=5, cmap='viridis', linewidth=0.1, ax=ax,
            edgecolor='white', legend=True, alpha=0.7)
    ax.set_axis_off()
    plt.title("Anomaly Scores")
    plt.savefig(name)
    plt.show()