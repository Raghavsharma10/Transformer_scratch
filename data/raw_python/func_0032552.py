def getFieldInfo(fieldnum):
    """Returns a dictionary containing the metadata of a K2 Campaign field.

    Raises a ValueError if the field number is unknown.

    Parameters
    ----------
    fieldnum : int
        Campaign field number (e.g. 0, 1, 2, ...)

    Returns
    -------
    field : dict
        The dictionary contains the keys
        'ra', 'dec', 'roll' (floats in decimal degrees),
        'start', 'stop', (strings in YYYY-MM-DD format)
        and 'comments' (free text).
    """
    try:
        info = _getCampaignDict()["c{0}".format(fieldnum)]
        # Print warning messages if necessary
        if "preliminary" in info and info["preliminary"] == "True":
            logger.warning("Warning: the position of field {0} is preliminary. "
                           "Do not use this position for your final "
                           "target selection!".format(fieldnum))
        return info
    except KeyError:
        raise ValueError("Field {0} not set in this version "
                         "of the code".format(fieldnum))