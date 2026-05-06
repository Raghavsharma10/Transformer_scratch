def heatmap(x, y, z, title="", cmap=plt.cm.YlOrRd, bins=20,
            xlim=(-250, 250), ylim=(422.5, -47.5),
            facecolor='lightgray', facecolor_alpha=0.4,
            court_color="black", court_lw=0.5, outer_lines=False,
            flip_court=False, ax=None, **kwargs):

    """
    Returns an AxesImage object that contains a heatmap.

    TODO: Redo some code and explain parameters
    """

    # Bin the FGA (x, y) and Calculcate the mean number of times shot was
    # made (z) within each bin
    # mean is the calculated FG percentage for each bin
    mean, xedges, yedges, binnumber = binned_statistic_2d(x=x, y=y,
                                                          values=z,
                                                          statistic='mean',
                                                          bins=bins)

    if ax is None:
        ax = plt.gca()

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    ax.patch.set_facecolor(facecolor)
    ax.patch.set_alpha(facecolor_alpha)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    heatmap = ax.imshow(mean.T, origin='lower', extent=[xedges[0], xedges[-1],
                        yedges[0], yedges[-1]], interpolation='nearest',
                        cmap=cmap)

    return heatmap