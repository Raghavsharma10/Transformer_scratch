def networks_distribution(df, filepath=None):
    """
    Generates two alternative plots describing the distribution of
    variables `mse` and `size`. It is intended to be used over a list
    of logical networks.


    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns `mse` and `size`

    filepath: str
        Absolute path to a folder where to write the plots


    Returns
    -------
    tuple
        Generated plots


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    df.mse = df.mse.map(lambda f: "%.4f" % f)

    g = sns.JointGrid(x="mse", y="size", data=df)

    g.plot_joint(sns.violinplot, scale='count')
    g.ax_joint.set_yticks(range(df['size'].min(), df['size'].max() + 1))
    g.ax_joint.set_yticklabels(range(df['size'].min(), df['size'].max() + 1))

    for tick in g.ax_joint.get_xticklabels():
        tick.set_rotation(90)

    g.ax_joint.set_xlabel("MSE")
    g.ax_joint.set_ylabel("Size")

    for i, t in enumerate(g.ax_joint.get_xticklabels()):
        c = df[df['mse'] == t.get_text()].shape[0]
        g.ax_marg_x.annotate(c, xy=(i, 0.5), va="center", ha="center", size=20, rotation=90)

    for i, t in enumerate(g.ax_joint.get_yticklabels()):
        s = int(t.get_text())
        c = df[df['size'] == s].shape[0]
        g.ax_marg_y.annotate(c, xy=(0.5, s), va="center", ha="center", size=20)

    if filepath:
        g.savefig(os.path.join(filepath, 'networks-distribution.pdf'))

    plt.figure()
    counts = df[["size", "mse"]].reset_index(level=0).groupby(["size", "mse"], as_index=False).count()
    cp = counts.pivot("size", "mse", "index").sort_index()

    ax = sns.heatmap(cp, annot=True, fmt=".0f", linewidths=.5)
    ax.set_xlabel("MSE")
    ax.set_ylabel("Size")

    if filepath:
        plt.savefig(os.path.join(filepath, 'networks-heatmap.pdf'))

    return g, ax