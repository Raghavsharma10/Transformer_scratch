def is_open_now(location=None, attr=None):
    """
    Returns False if the location is closed, or the OpeningHours object
    to show the location is currently open.
    Same as `is_open` but passes `now` to `utils.is_open` to bypass `get_now()`. 
    """
    obj = utils.is_open(location, now=datetime.datetime.now())
    if obj is False:
        return False
    if attr is not None:
        return getattr(obj, attr)
    return obj