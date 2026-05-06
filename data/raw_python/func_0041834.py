def label_subplot(ax=None, x=0.5, y=-0.25, text="(a)", **kwargs):
    """Create a subplot label."""
    if ax is None:
        ax = plt.gca()
    ax.text(x=x, y=y, s=text, transform=ax.transAxes,
            horizontalalignment="center", verticalalignment="top", **kwargs)