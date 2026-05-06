def behaviors_distribution(df, filepath=None):
    """
    Plots the distribution of logical networks across input-output behaviors.
    Optionally, input-output behaviors can be grouped by MSE.

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns `networks` and optionally `mse`

    filepath: str
        Absolute path to a folder where to write the plot

    Returns
    -------
    plot
        Generated plot


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    cols = ["networks", "index"]
    rcols = ["Logical networks", "Input-Output behaviors"]
    sort_cols = ["networks"]

    if "mse" in df.columns:
        cols.append("mse")
        rcols.append("MSE")
        sort_cols = ["mse"] + sort_cols

        df.mse = df.mse.map(lambda f: "%.4f" % f)

    df = df.sort_values(sort_cols).reset_index(drop=True).reset_index(level=0)[cols]
    df.columns = rcols

    if "MSE" in df.columns:
        g = sns.factorplot(x='Input-Output behaviors', y='Logical networks', hue='MSE', data=df, aspect=3, kind='bar', legend_out=False)
    else:
        g = sns.factorplot(x='Input-Output behaviors', y='Logical networks', data=df, aspect=3, kind='bar', legend_out=False)

    g.ax.set_xticks([])
    if filepath:
        g.savefig(os.path.join(filepath, 'behaviors-distribution.pdf'))

    return g