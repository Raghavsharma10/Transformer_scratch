def create_context_plot(ra, dec, name="Your object"):
    """Creates a K2FootprintPlot showing a given position in context
    with respect to the campaigns."""
    plot = K2FootprintPlot()
    plot.plot_galactic()
    plot.plot_ecliptic()
    for c in range(0, 20):
        plot.plot_campaign_outline(c, facecolor="#666666")
    # for c in [11, 12, 13, 14, 15, 16]:
    #    plot.plot_campaign_outline(c, facecolor="green")
    plot.ax.scatter(ra, dec, marker='x', s=250, lw=3, color="red", zorder=500)
    plot.ax.text(ra, dec - 2, name,
                 ha="center", va="top", color="red",
                 fontsize=20, fontweight='bold', zorder=501)
    return plot