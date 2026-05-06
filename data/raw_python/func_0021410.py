def length_plots(array, name, path, title=None, n50=None, color="#4CB391", figformat="png"):
    """Create histogram of normal and log transformed read lengths."""
    logging.info("Nanoplotter: Creating length plots for {}.".format(name))
    maxvalx = np.amax(array)
    if n50:
        logging.info("Nanoplotter: Using {} reads with read length N50 of {}bp and maximum of {}bp."
                     .format(array.size, n50, maxvalx))
    else:
        logging.info("Nanoplotter: Using {} reads maximum of {}bp.".format(array.size, maxvalx))

    plots = []
    HistType = namedtuple('HistType', 'weight name ylabel')
    for h_type in [HistType(None, "", "Number of reads"),
                   HistType(array, "Weighted ", "Number of bases")]:
        histogram = Plot(
            path=path + h_type.name.replace(" ", "_") + "Histogram"
            + name.replace(' ', '') + "." + figformat,
            title=h_type.name + "Histogram of read lengths")
        ax = sns.distplot(
            a=array,
            kde=False,
            hist=True,
            bins=max(round(int(maxvalx) / 500), 10),
            color=color,
            hist_kws=dict(weights=h_type.weight,
                          edgecolor=color,
                          linewidth=0.2,
                          alpha=0.8))
        if n50:
            plt.axvline(n50)
            plt.annotate('N50', xy=(n50, np.amax([h.get_height() for h in ax.patches])), size=8)
        ax.set(
            xlabel='Read length',
            ylabel=h_type.ylabel,
            title=title or histogram.title)
        plt.ticklabel_format(style='plain', axis='y')
        histogram.fig = ax.get_figure()
        histogram.save(format=figformat)
        plt.close("all")

        log_histogram = Plot(
            path=path + h_type.name.replace(" ", "_") + "LogTransformed_Histogram"
            + name.replace(' ', '') + "." + figformat,
            title=h_type.name + "Histogram of read lengths after log transformation")
        ax = sns.distplot(
            a=np.log10(array),
            kde=False,
            hist=True,
            color=color,
            hist_kws=dict(weights=h_type.weight,
                          edgecolor=color,
                          linewidth=0.2,
                          alpha=0.8))
        ticks = [10**i for i in range(10) if not 10**i > 10 * maxvalx]
        ax.set(
            xticks=np.log10(ticks),
            xticklabels=ticks,
            xlabel='Read length',
            ylabel=h_type.ylabel,
            title=title or log_histogram.title)
        if n50:
            plt.axvline(np.log10(n50))
            plt.annotate('N50', xy=(np.log10(n50), np.amax(
                [h.get_height() for h in ax.patches])), size=8)
        plt.ticklabel_format(style='plain', axis='y')
        log_histogram.fig = ax.get_figure()
        log_histogram.save(format=figformat)
        plt.close("all")
        plots.extend([histogram, log_histogram])
    plots.append(yield_by_minimal_length_plot(array=array,
                                              name=name,
                                              path=path,
                                              title=title,
                                              color=color,
                                              figformat=figformat))
    return plots