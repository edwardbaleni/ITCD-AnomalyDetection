import matplotlib.pylab as plt
def plot(erf, normal, anomaly):
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    erf.plot.imshow(ax=ax)
    normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
    anomaly.plot(ax=ax, facecolor = 'none',edgecolor='blue')