def experimental_designs(df, filepath=None):
    """
    For each experimental design it plot all the corresponding
    experimental conditions in a different plot

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns `id` and starting with `TR:`

    filepath: str
        Absolute path to a folder where to write the plot


    Returns
    -------
    list
        Generated plots


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    axes = []
    bw = matplotlib.colors.ListedColormap(['white', 'black'])
    cols = df.columns
    for i, dd in df.groupby("id"):
        cues = dd.drop([c for c in cols if not c.startswith("TR:")] + ["id"], axis=1).reset_index(drop=True)
        cues.columns = [c[3:] for c in cues.columns]

        plt.figure(figsize=(max((len(cues.columns)-1) * .5, 4), max(len(cues)*0.6, 2.5)))

        ax = sns.heatmap(cues, linewidths=.5, cbar=False, cmap=bw, linecolor='gray')
        _ = [t.set_color('r') if t.get_text().endswith('i') else t.set_color('g') for t in ax.xaxis.get_ticklabels()]

        ax.set_xlabel("Stimuli (green) and Inhibitors (red)")
        ax.set_ylabel("Experimental condition")

        plt.tight_layout()
        axes.append(ax)

        if filepath:
            plt.savefig(os.path.join(filepath, 'design-%s.pdf' % i))

    return axes