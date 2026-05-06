def violinplot(x=None, y=None, data=None, bw=0.2, scale='width',
               inner=None, ax=None, **kwargs):
    """Wrapper around Seaborn's Violinplot specifically for [0, 1] ranged data

    What's different:
    - bw = 0.2: Sets bandwidth to be small and the same between datasets
    - scale = 'width': Sets the width of all violinplots to be the same
    - inner = None: Don't plot a boxplot or points inside the violinplot
    """
    if ax is None:
        ax = plt.gca()

    sns.violinplot(x, y, data=data, bw=bw, scale=scale, inner=inner, ax=ax,
                   **kwargs)
    ax.set(ylim=(0, 1), yticks=(0, 0.5, 1))
    return ax