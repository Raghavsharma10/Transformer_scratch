def float_value_convert(dictin, dropfailedvalues=False):
    # type: (DictUpperBound, bool) -> Dict
    """Convert values of dictionary to floats

    Args:
        dictin (DictUpperBound): Input dictionary
        dropfailedvalues (bool): Whether to drop dictionary entries where key conversion fails. Defaults to False.

    Returns:
        Dict: Dictionary with values converted to floats

    """
    return key_value_convert(dictin, valuefn=float, dropfailedvalues=dropfailedvalues)