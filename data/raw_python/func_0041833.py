def set_sns(style="white", context="paper", font_scale=1.5, color_codes=True,
            rc={}):
    """Set default plot style using seaborn.

    Font size is set to match the size of the tick labels, rather than the axes
    labels.
    """
    rcd = {"lines.markersize": 8, "lines.markeredgewidth": 1.25,
           "legend.fontsize": "small", "font.size": 12/1.5*font_scale,
           "legend.frameon": True, "axes.formatter.limits": (-5, 5),
           "axes.grid": True}
    rcd.update(rc)
    import seaborn as sns
    sns.set(style=style, context=context, font_scale=font_scale,
            color_codes=color_codes, rc=rcd)