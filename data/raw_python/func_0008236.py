def get_root(w):
    """
    Simple method to access root for a widget
    """
    next_level = w
    while next_level.master:
        next_level = next_level.master
    return next_level