def update(d, e):
    """Return a copy of dict `d` updated with dict `e`."""
    res = copy.copy(d)
    res.update(e)
    return res