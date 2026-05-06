def scatter(x, y, names, path, plots, color="#4CB391", figformat="png",
            stat=None, log=False, minvalx=0, minvaly=0, title=None, plot_settings=None):
    """Create bivariate plots.

    Create four types of bivariate plots of x vs y, containing marginal summaries
    -A scatter plot with histograms on axes
    -A hexagonal binned plot with histograms on axes
    -A kernel density plot with density curves on axes
    -A pauvre-style plot using code from https://github.com/conchoecia/pauvre
    """
    logging.info("Nanoplotter: Creating {} vs {} plots using statistics from {} reads.".format(
        names[0], names[1], x.size))
    if not contains_variance([x, y], names):
        return []
    sns.set(style="ticks", **plot_settings)
    maxvalx = np.amax(x)
    maxvaly = np.amax(y)

    plots_made = []

    if plots["hex"]:
        hex_plot = Plot(
            path=path + "_hex." + figformat,
            title="{} vs {} plot using hexagonal bins".format(names[0], names[1]))
        plot = sns.jointplot(
            x=x,
            y=y,
            kind="hex",
            color=color,
            stat_func=stat,
            space=0,
            xlim=(minvalx, maxvalx),
            ylim=(minvaly, maxvaly),
            height=10)
        plot.set_axis_labels(names[0], names[1])
        if log:
            hex_plot.title = hex_plot.title + " after log transformation of read lengths"
            ticks = [10**i for i in range(10) if not 10**i > 10 * (10**maxvalx)]
            plot.ax_joint.set_xticks(np.log10(ticks))
            plot.ax_marg_x.set_xticks(np.log10(ticks))
            plot.ax_joint.set_xticklabels(ticks)
        plt.subplots_adjust(top=0.90)
        plot.fig.suptitle(title or "{} vs {} plot".format(names[0], names[1]), fontsize=25)
        hex_plot.fig = plot
        hex_plot.save(format=figformat)
        plots_made.append(hex_plot)

    sns.set(style="darkgrid", **plot_settings)
    if plots["dot"]:
        dot_plot = Plot(
            path=path + "_dot." + figformat,
            title="{} vs {} plot using dots".format(names[0], names[1]))
        plot = sns.jointplot(
            x=x,
            y=y,
            kind="scatter",
            color=color,
            stat_func=stat,
            xlim=(minvalx, maxvalx),
            ylim=(minvaly, maxvaly),
            space=0,
            height=10,
            joint_kws={"s": 1})
        plot.set_axis_labels(names[0], names[1])
        if log:
            dot_plot.title = dot_plot.title + " after log transformation of read lengths"
            ticks = [10**i for i in range(10) if not 10**i > 10 * (10**maxvalx)]
            plot.ax_joint.set_xticks(np.log10(ticks))
            plot.ax_marg_x.set_xticks(np.log10(ticks))
            plot.ax_joint.set_xticklabels(ticks)
        plt.subplots_adjust(top=0.90)
        plot.fig.suptitle(title or "{} vs {} plot".format(names[0], names[1]), fontsize=25)
        dot_plot.fig = plot
        dot_plot.save(format=figformat)
        plots_made.append(dot_plot)

    if plots["kde"]:
        idx = np.random.choice(x.index, min(2000, len(x)), replace=False)
        kde_plot = Plot(
            path=path + "_kde." + figformat,
            title="{} vs {} plot using a kernel density estimation".format(names[0], names[1]))
        plot = sns.jointplot(
            x=x[idx],
            y=y[idx],
            kind="kde",
            clip=((0, np.Inf), (0, np.Inf)),
            xlim=(minvalx, maxvalx),
            ylim=(minvaly, maxvaly),
            space=0,
            color=color,
            stat_func=stat,
            shade_lowest=False,
            height=10)
        plot.set_axis_labels(names[0], names[1])
        if log:
            kde_plot.title = kde_plot.title + " after log transformation of read lengths"
            ticks = [10**i for i in range(10) if not 10**i > 10 * (10**maxvalx)]
            plot.ax_joint.set_xticks(np.log10(ticks))
            plot.ax_marg_x.set_xticks(np.log10(ticks))
            plot.ax_joint.set_xticklabels(ticks)
        plt.subplots_adjust(top=0.90)
        plot.fig.suptitle(title or "{} vs {} plot".format(names[0], names[1]), fontsize=25)
        kde_plot.fig = plot
        kde_plot.save(format=figformat)
        plots_made.append(kde_plot)

    if plots["pauvre"] and names == ['Read lengths', 'Average read quality'] and log is False:
        pauvre_plot = Plot(
            path=path + "_pauvre." + figformat,
            title="{} vs {} plot using pauvre-style @conchoecia".format(names[0], names[1]))
        sns.set(style="white", **plot_settings)
        margin_plot(df=pd.DataFrame({"length": x, "meanQual": y}),
                    Y_AXES=False,
                    title=title or "Length vs Quality in Pauvre-style",
                    plot_maxlen=None,
                    plot_minlen=0,
                    plot_maxqual=None,
                    plot_minqual=0,
                    lengthbin=None,
                    qualbin=None,
                    BASENAME="whatever",
                    path=pauvre_plot.path,
                    fileform=[figformat],
                    dpi=600,
                    TRANSPARENT=True,
                    QUIET=True)
        plots_made.append(pauvre_plot)
    plt.close("all")
    return plots_made