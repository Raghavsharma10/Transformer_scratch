def mappings_frequency(df, filepath=None):
    """
    Plots the frequency of logical conjunction mappings

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns `frequency` and `mapping`

    filepath: str
        Absolute path to a folder where to write the plot


    Returns
    -------
    plot
        Generated plot


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    df = df.sort_values('frequency')
    df['conf'] = df.frequency.map(lambda f: 0 if f < 0.2 else 1 if f < 0.8 else 2)

    g = sns.factorplot(x="mapping", y="frequency", data=df, aspect=3, hue='conf', legend=False)
    for tick in g.ax.get_xticklabels():
        tick.set_rotation(90)

    g.ax.set_ylim([-.05, 1.05])

    g.ax.set_xlabel("Logical mapping")
    g.ax.set_ylabel("Frequency")

    if filepath:
        g.savefig(os.path.join(filepath, 'mappings-frequency.pdf'))

    return g