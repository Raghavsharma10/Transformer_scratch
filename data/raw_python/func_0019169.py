def is_open(location=None, attr=None):
    """
    Returns False if the location is closed, or the OpeningHours object
    to show the location is currently open.
    """
    obj = utils.is_open(location)
    if obj is False:
        return False
    if attr is not None:
        return getattr(obj, attr)
    return obj