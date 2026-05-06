def _get_color_list():
    """Get cycle of colors in a way compatible with all matplotlib versions"""
    if 'axes.prop_cycle' in plt.rcParams:
        return [p['color'] for p in list(plt.rcParams['axes.prop_cycle'])]
    else:
        return plt.rcParams['axes.color_cycle']