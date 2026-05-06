def from_rgb(r, g=None, b=None):
    """
    Return the nearest xterm 256 color code from rgb input.
    """
    c = r if isinstance(r, list) else [r, g, b]
    best = {}

    for index, item in enumerate(colors):
        d = __distance(item, c)
        if(not best or d <= best['distance']):
            best = {'distance': d, 'index': index}

    if 'index' in best:
        return best['index']
    else:
        return 1