def stripboxplot(x, y, data, ax=None, significant=None, **kwargs):
    """
    Overlay a stripplot on top of a boxplot.
    """
    ax = sb.boxplot(
        x=x,
        y=y,
        data=data,
        ax=ax,
        fliersize=0,
        **kwargs
    )

    plot = sb.stripplot(
        x=x,
        y=y,
        data=data,
        ax=ax,
        jitter=kwargs.pop("jitter", 0.05),
        color=kwargs.pop("color", "0.3"),
        **kwargs
    )

    if data[y].min() >= 0:
        hide_negative_y_ticks(plot)
    if significant is not None:
        add_significance_indicator(plot=plot, significant=significant)

    return plot