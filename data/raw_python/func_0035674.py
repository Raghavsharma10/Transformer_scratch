def desaturate(c, k=0):
    """
    Utility function to desaturate a color c by an amount k.
    """
    from matplotlib.colors import ColorConverter
    c = ColorConverter().to_rgb(c)
    intensity = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return [intensity * k + i * (1 - k) for i in c]