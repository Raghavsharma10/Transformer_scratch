def list_fonts():
    """List system fonts

    Returns
    -------
    fonts : list of str
        List of system fonts.
    """
    vals = _list_fonts()
    for font in _vispy_fonts:
        vals += [font] if font not in vals else []
    vals = sorted(vals, key=lambda s: s.lower())
    return vals