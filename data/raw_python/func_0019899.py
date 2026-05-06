def plotcorr(X, plotargs=None, full=True, labels=None):
    """
    Plots a scatterplot matrix of subplots.  
    
    Usage:
    
        plotcorr(X)
        
        plotcorr(..., plotargs=...)  # e.g., 'r*', 'bo', etc.
        
        plotcorr(..., full=...)  # e.g., True or False
        
        plotcorr(..., labels=...)  # e.g., ['label1', 'label2', ...]

    Each column of "X" is plotted against other columns, resulting in
    a ncols by ncols grid of subplots with the diagonal subplots labeled 
    with "labels".  "X" is an array of arrays (i.e., a 2d matrix), a 1d array
    of MCERP.UncertainFunction/Variable objects, or a mixture of the two.
    Additional keyword arguments are passed on to matplotlib's "plot" command. 
    Returns the matplotlib figure object containing the subplot grid.
    """
    import matplotlib.pyplot as plt

    X = [Xi._mcpts if isinstance(Xi, UncertainFunction) else Xi for Xi in X]
    X = np.atleast_2d(X)
    numvars, numdata = X.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if full:
            if ax.is_first_col():
                ax.yaxis.set_ticks_position("left")
            if ax.is_last_col():
                ax.yaxis.set_ticks_position("right")
            if ax.is_first_row():
                ax.xaxis.set_ticks_position("top")
            if ax.is_last_row():
                ax.xaxis.set_ticks_position("bottom")
        else:
            if ax.is_first_row():
                ax.xaxis.set_ticks_position("top")
            if ax.is_last_col():
                ax.yaxis.set_ticks_position("right")

    # Label the diagonal subplots...
    if not labels:
        labels = ["x" + str(i) for i in range(numvars)]

    for i, label in enumerate(labels):
        axes[i, i].annotate(
            label, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center"
        )

    # Plot the data
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        if full:
            idx = [(i, j), (j, i)]
        else:
            idx = [(i, j)]
        for x, y in idx:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            if plotargs is None:
                if len(X[x]) > 100:
                    plotargs = ",b"  # pixel marker
                else:
                    plotargs = ".b"  # point marker
            axes[x, y].plot(X[y], X[x], plotargs)
            ylim = min(X[y]), max(X[y])
            xlim = min(X[x]), max(X[x])
            axes[x, y].set_ylim(
                xlim[0] - (xlim[1] - xlim[0]) * 0.1, xlim[1] + (xlim[1] - xlim[0]) * 0.1
            )
            axes[x, y].set_xlim(
                ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1] + (ylim[1] - ylim[0]) * 0.1
            )

    # Turn on the proper x or y axes ticks.
    if full:
        for i, j in zip(list(range(numvars)), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            axes[i, j].yaxis.set_visible(True)
    else:
        for i in range(numvars - 1):
            axes[0, i + 1].xaxis.set_visible(True)
            axes[i, -1].yaxis.set_visible(True)
        for i in range(1, numvars):
            for j in range(0, i):
                fig.delaxes(axes[i, j])

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars % 2:
        xlimits = axes[0, -1].get_xlim()
        ylimits = axes[-1, 0].get_ylim()
        axes[-1, -1].set_xlim(xlimits)
        axes[-1, -1].set_ylim(ylimits)

    return fig