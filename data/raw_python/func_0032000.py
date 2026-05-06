def plot_h(data, cols, wspace=.1, plot_kw=None, **kwargs):
    """
    Plot horizontally

    Args:
        data: DataFrame of data
        cols: columns to be plotted
        wspace: spacing between plots
        plot_kw: kwargs for each plot
        **kwargs: kwargs for the whole plot

    Returns:
        axes for plots

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> idx = range(5)
        >>> data = pd.DataFrame(dict(a=np.exp(idx), b=idx), index=idx)
        >>> # plot_h(data=data, cols=['a', 'b'], wspace=.2, plot_kw=[dict(style='.-'), dict()])
    """
    import matplotlib.pyplot as plt

    if plot_kw is None: plot_kw = [dict()] * len(cols)

    _, axes = plt.subplots(nrows=1, ncols=len(cols), **kwargs)
    plt.subplots_adjust(wspace=wspace)
    for n, col in enumerate(cols):
        data.loc[:, col].plot(ax=axes[n], **plot_kw[n])

    return axes