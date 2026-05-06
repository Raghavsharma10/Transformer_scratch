def roc_curve(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8),
              xlabel="Probability of False Detection",
              ylabel="Probability of Detection",
              title="ROC Curve", ticks=np.arange(0, 1.1, 0.1), dpi=300,
              legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5),
              label_fontsize=14, title_fontsize=16, tick_fontsize=12):
    """
    Plots a set receiver/relative operating characteristic (ROC) curves from DistributedROC objects.

    The ROC curve shows how well a forecast discriminates between two outcomes over a series of thresholds. It
    features Probability of Detection (True Positive Rate) on the y-axis and Probability of False Detection
    (False Alarm Rate) on the x-axis. This plotting function allows one to customize the colors and markers of the
    ROC curves as well as the parameters of the legend and the title.

    Args:
        roc_objs (list): DistributedROC objects being plotted.
        obj_labels (list): Label describing the forecast associated with a DistributedROC object.
        colors (list): List of matplotlib-readable colors (names or hex-values) for each curve.
        markers (list): Matplotlib marker (e.g. *, o, v, etc.) for each curve.
        filename (str): Name of figure file being saved.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): The title of the figure.
        ticks (numpy.ndarray): Values shown on the x and y axes.
        dpi (int): Figure resolution in dots per inch.
        legend_params (None, dict): Keyword arguments for the formatting of the figure legend.
        bootstrap_sets (list): List of lists of DistributedROC objects that were bootstrap resampled for each model.
        ci (tuple of 2 floats): Quantiles of the edges of the bootstrap confidence intervals ranging from 0 to 100.
        label_fontsize (int): Font size of the x and y axis labels.
        title_fontsize (int): Font size of the title.
        tick_fontsize (int): Font size of the x and y tick labels.

    Examples:

        >>> from hagelslag.evaluation import DistributedROC
        >>> import numpy as np
        >>> forecasts = np.random.random(1000)
        >>> obs = np.random.random_integers(0, 1, 1000)
        >>> roc = DistributedROC()
        >>> roc.update(forecasts, obs)
        >>> roc_curve([roc], ["Random"], ["orange"], ["o"], "random_roc.png")
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=12, framealpha=1, frameon=True)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(ticks, ticks, "k--")
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            broc_curves = np.dstack([b_roc.roc_curve().values for b_roc in b_set])
            pod_range = np.percentile(broc_curves[:,0], ci, axis=1)
            pofd_range = np.percentile(broc_curves[:, 1], ci, axis=1)
            pod_poly = np.concatenate((pod_range[1], pod_range[0, ::-1]))
            pofd_poly = np.concatenate((pofd_range[0], pofd_range[1, ::-1]))
            pod_poly[np.isnan(pod_poly)] = 0
            pofd_poly[np.isnan(pofd_poly)] = 0
            plt.fill(pofd_poly, pod_poly, alpha=0.5, color=colors[b])
    for r, roc_obj in enumerate(roc_objs):
        roc_data = roc_obj.roc_curve()
        plt.plot(roc_data["POFD"], roc_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(ticks, fontsize=tick_fontsize)
    plt.yticks(ticks, fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()