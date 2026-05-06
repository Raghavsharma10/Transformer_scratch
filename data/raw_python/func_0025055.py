def reliability_diagram(rel_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Forecast Probability",
                        ylabel="Observed Relative Frequency", ticks=np.arange(0, 1.05, 0.05), dpi=300, inset_size=1.5,
                        title="Reliability Diagram", legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5)):
    """
    Plot reliability curves against a 1:1 diagonal to determine if probability forecasts are consistent with their
    observed relative frequency.

    Args:
        rel_objs (list): List of DistributedReliability objects.
        obj_labels (list): List of labels describing the forecast model associated with each curve.
        colors (list): List of colors for each line
        markers (list): List of line markers
        filename (str): Where to save the figure.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        ticks (array): Tick value labels for the x and y axes.
        dpi (int): resolution of the saved figure in dots per inch.
        inset_size (float): Size of inset
        title (str): Title of figure
        legend_params (dict): Keyword arguments for the plot legend.
        bootstrap_sets (list): A list of arrays of bootstrapped DistributedROC objects. If not None,
            confidence regions will be plotted.
        ci (tuple): tuple of bootstrap confidence interval percentiles
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=10, framealpha=1, frameon=True)
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    inset_hist = inset_axes(ax, width=inset_size, height=inset_size, loc=2)
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            brel_curves = np.dstack([b_rel.reliability_curve().values for b_rel in b_set])
            bin_range = np.percentile(brel_curves[:,0], ci, axis=1)
            rel_range = np.percentile(brel_curves[:, 3], ci, axis=1)
            bin_poly = np.concatenate((bin_range[1], bin_range[0, ::-1]))
            rel_poly = np.concatenate((rel_range[1], rel_range[0, ::-1]))
            bin_poly[np.isnan(bin_poly)] = 0
            rel_poly[np.isnan(rel_poly)] = 0
            plt.fill(bin_poly, rel_poly, alpha=0.5, color=colors[b])
    for r, rel_obj in enumerate(rel_objs):
        rel_curve = rel_obj.reliability_curve()
        ax.plot(rel_curve["Bin_Start"], rel_curve["Positive_Relative_Freq"], color=colors[r], marker=markers[r],
                 label=obj_labels[r])
        inset_hist.semilogy(rel_curve["Bin_Start"], rel_curve["Total_Relative_Freq"], color=colors[r],
                            marker=markers[r])
        inset_hist.set_xlabel("Forecast Probability")
        inset_hist.set_ylabel("Forecast Relative Frequency")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend(**legend_params)
    ax.set_title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()