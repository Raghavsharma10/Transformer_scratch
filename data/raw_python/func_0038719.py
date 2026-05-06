def differences_distribution(df, filepath=None):
    """
    For each experimental design it plot all the corresponding
    generated differences in different plots

    Parameters
    ----------
    df: `pandas.DataFrame`_
        DataFrame with columns `id`, `pairs`, and starting with `DIF:`

    filepath: str
        Absolute path to a folder where to write the plots


    Returns
    -------
    list
        Generated plots


    .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    """

    axes = []
    cols = df.columns
    for i, dd in df.groupby("id"):
        palette = sns.color_palette("Set1", len(dd))
        plt.figure()

        readouts = dd.drop([c for c in cols if not c.startswith("DIF:")] + ["id"], axis=1).reset_index(drop=True)
        readouts.columns = [c[4:] for c in readouts.columns]

        ax1 = readouts.T.plot(kind='bar', stacked=True, color=palette)

        ax1.set_xlabel("Readout")
        ax1.set_ylabel("Pairwise differences")
        plt.tight_layout()

        if filepath:
            plt.savefig(os.path.join(filepath, 'design-%s-readouts.pdf' % i))

        plt.figure()
        behaviors = dd[["pairs"]].reset_index(drop=True)
        ax2 = behaviors.plot.bar(color=palette, legend=False)

        ax2.set_xlabel("Experimental condition")
        ax2.set_ylabel("Pairs of input-output behaviors")
        plt.tight_layout()

        if filepath:
            plt.savefig(os.path.join(filepath, 'design-%s-behaviors.pdf' % i))

        axes.append((ax1, ax2))

    return axes