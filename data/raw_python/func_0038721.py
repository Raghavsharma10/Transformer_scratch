def intervention_strategies(df, filepath=None):
    """
    Plots all intervention strategies

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns starting with `TR:`

    filepath: str
        Absolute path to a folder where to write the plot


    Returns
    -------
    plot
        Generated plot


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """
    logger = logging.getLogger("caspo")

    LIMIT = 50
    if len(df) > LIMIT:
        msg = "Too many intervention strategies to visualize. A sample of %s strategies will be considered." % LIMIT
        logger.warning(msg)
        df = df.sample(LIMIT)

    values = np.unique(df.values.flatten())
    if len(values) == 3:
        rwg = matplotlib.colors.ListedColormap(['red', 'white', 'green'])
    elif 1 in values:
        rwg = matplotlib.colors.ListedColormap(['white', 'green'])
    else:
        rwg = matplotlib.colors.ListedColormap(['red', 'white'])

    plt.figure(figsize=(max((len(df.columns)-1) * .5, 4), max(len(df)*0.6, 2.5)))

    df.columns = [c[3:] for c in df.columns]
    ax = sns.heatmap(df, linewidths=.5, cbar=False, cmap=rwg, linecolor='gray')

    ax.set_xlabel("Species")
    ax.set_ylabel("Intervention strategy")

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.tight_layout()

    if filepath:
        plt.savefig(os.path.join(filepath, 'strategies.pdf'))

    return ax