def integer_value_convert(dictin, dropfailedvalues=False):
    # type: (DictUpperBound, bool) -> Dict
    """Convert values of dictionary to integers

    Args:
        dictin (DictUpperBound): Input dictionary
        dropfailedvalues (bool): Whether to drop dictionary entries where key conversion fails. Defaults to False.

    Returns:
        Dict: Dictionary with values converted to integers

    """
    return key_value_convert(dictin, valuefn=int, dropfailedvalues=dropfailedvalues)