def avg_dicts(dictin1, dictin2, dropmissing=True):
    # type: (DictUpperBound, DictUpperBound, bool) -> Dict
    """Create a new dictionary from two dictionaries by averaging values

    Args:
        dictin1 (DictUpperBound): First input dictionary
        dictin2 (DictUpperBound): Second input dictionary
        dropmissing (bool): Whether to drop keys missing in one dictionary. Defaults to True.

    Returns:
        Dict: Dictionary with values being average of 2 input dictionaries

    """
    dictout = dict()
    for key in dictin1:
        if key in dictin2:
            dictout[key] = (dictin1[key] + dictin2[key]) / 2
        elif not dropmissing:
            dictout[key] = dictin1[key]
    if not dropmissing:
        for key in dictin2:
            if key not in dictin1:
                dictout[key] = dictin2[key]
    return dictout