def create_context_plot_zoomed(ra, dec, name="Your object", size=3):
    """Creates a K2FootprintPlot showing a given position in context
    with respect to the campaigns."""
    plot = K2FootprintPlot(figsize=(8, 8))
    for c in range(0, 20):
        plot.plot_campaign(c)
    plot.ax.scatter(ra, dec, marker='x', s=250, lw=3, color="red", zorder=500)
    plot.ax.text(ra, dec - 0.05*size, name,
                 ha="center", va="top", color="red",
                 fontsize=20, fontweight='bold', zorder=501)
    plot.ax.set_xlim([ra - size/2., ra + size/2.])
    plot.ax.set_ylim([dec - size/2., dec + size/2.])
    return plot