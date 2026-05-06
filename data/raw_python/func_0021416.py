def violin_or_box_plot(df, y, figformat, path, y_name,
                       title=None, plot="violin", log=False, palette=None):
    """Create a violin or boxplot from the received DataFrame.

    The x-axis should be divided based on the 'dataset' column,
    the y-axis is specified in the arguments
    """
    comp = Plot(path=path + "NanoComp_" + y.replace(' ', '_') + '.' + figformat,
                title="Comparing {}".format(y))
    if y == "quals":
        comp.title = "Comparing base call quality scores"

    if plot == 'violin':
        logging.info("Nanoplotter: Creating violin plot for {}.".format(y))
        process_violin_and_box(ax=sns.violinplot(x="dataset",
                                                 y=y,
                                                 data=df,
                                                 inner=None,
                                                 cut=0,
                                                 palette=palette,
                                                 linewidth=0),
                               log=log,
                               plot_obj=comp,
                               title=title,
                               y_name=y_name,
                               figformat=figformat,
                               ymax=np.amax(df[y]))
    elif plot == 'box':
        logging.info("Nanoplotter: Creating box plot for {}.".format(y))
        process_violin_and_box(ax=sns.boxplot(x="dataset",
                                              y=y,
                                              data=df,
                                              palette=palette),
                               log=log,
                               plot_obj=comp,
                               title=title,
                               y_name=y_name,
                               figformat=figformat,
                               ymax=np.amax(df[y]))
    elif plot == 'ridge':
        logging.info("Nanoplotter: Creating ridges plot for {}.".format(y))
        comp.fig, axes = joypy.joyplot(df,
                                       by="dataset",
                                       column=y,
                                       title=title or comp.title,
                                       x_range=[-0.05, np.amax(df[y])])
        if log:
            xticks = [float(i.get_text()) for i in axes[-1].get_xticklabels()]
            axes[-1].set_xticklabels([10**i for i in xticks])
        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=30, ha='center')
        comp.save(format=figformat)
    else:
        logging.error("Unknown comp plot type {}".format(plot))
        sys.exit("Unknown comp plot type {}".format(plot))
    plt.close("all")
    return [comp]