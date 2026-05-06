def predictions_variance(df, filepath=None):
    """
    Plots the mean variance prediction for each readout

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns starting with `VAR:`

    filepath: str
        Absolute path to a folder where to write the plots


    Returns
    -------
    plot
        Generated plot


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    df = df.filter(regex="^VAR:")

    by_readout = df.mean(axis=0).reset_index(level=0)
    by_readout.columns = ['Readout', 'Prediction variance (mean)']

    by_readout['Readout'] = by_readout.Readout.map(lambda n: n[4:])

    g1 = sns.factorplot(x='Readout', y='Prediction variance (mean)', data=by_readout, kind='bar', aspect=2)

    for tick in g1.ax.get_xticklabels():
        tick.set_rotation(90)

    if filepath:
        g1.savefig(os.path.join(filepath, 'predictions-variance.pdf'))

    return g1