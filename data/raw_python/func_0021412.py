def spatial_heatmap(array, path, title=None, color="Greens", figformat="png"):
    """Taking channel information and creating post run channel activity plots."""
    logging.info("Nanoplotter: Creating heatmap of reads per channel using {} reads."
                 .format(array.size))
    activity_map = Plot(
        path=path + "." + figformat,
        title="Number of reads generated per channel")
    layout = make_layout(maxval=np.amax(array))
    valueCounts = pd.value_counts(pd.Series(array))
    for entry in valueCounts.keys():
        layout.template[np.where(layout.structure == entry)] = valueCounts[entry]
    plt.figure()
    ax = sns.heatmap(
        data=pd.DataFrame(layout.template, index=layout.yticks, columns=layout.xticks),
        xticklabels="auto",
        yticklabels="auto",
        square=True,
        cbar_kws={"orientation": "horizontal"},
        cmap=color,
        linewidths=0.20)
    ax.set_title(title or activity_map.title)
    activity_map.fig = ax.get_figure()
    activity_map.save(format=figformat)
    plt.close("all")
    return [activity_map]