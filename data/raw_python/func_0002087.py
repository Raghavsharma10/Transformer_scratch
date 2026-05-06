def add_ticks_to_x(ax, newticks, newnames):
    """Add new ticks to an axis.

    I use this for the right-hand plotting of resonance names in my plots.
    """
    ticks = list(ax.get_xticks())
    ticks.extend(newticks)
    ax.set_xticks(ticks)

    names = list(ax.get_xticklabels())
    names.extend(newnames)
    ax.set_xticklabels(names)