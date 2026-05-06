def moothedata(data, key=None):
    """Return an amusing picture containing an item from a dict.

    Parameters
    ----------
    data: mapping
        A mapping, such as a raster dataset's ``meta`` or ``profile``
        property.
    key:
        A key of the ``data`` mapping.
    """
    if not key:
        key = choice(list(data.keys()))
        logger.debug("Using randomly chosen key: %s", key)
    msg = cow.Moose().milk("{0}: {1}".format(key.capitalize(), data[key]))
    return msg