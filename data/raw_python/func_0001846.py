def key_value_convert(dictin, keyfn=lambda x: x, valuefn=lambda x: x, dropfailedkeys=False, dropfailedvalues=False,
                      exception=ValueError):
    # type: (DictUpperBound, Callable[[Any], Any], Callable[[Any], Any], bool, bool, ExceptionUpperBound) -> Dict
    """Convert keys and/or values of dictionary using functions passed in as parameters

    Args:
        dictin (DictUpperBound): Input dictionary
        keyfn (Callable[[Any], Any]): Function to convert keys. Defaults to lambda x: x
        valuefn (Callable[[Any], Any]): Function to convert values. Defaults to lambda x: x
        dropfailedkeys (bool): Whether to drop dictionary entries where key conversion fails. Defaults to False.
        dropfailedvalues (bool): Whether to drop dictionary entries where value conversion fails. Defaults to False.
        exception (ExceptionUpperBound): The exception to expect if keyfn or valuefn fail. Defaults to ValueError.

    Returns:
        Dict: Dictionary with converted keys and/or values

    """
    dictout = dict()
    for key in dictin:
        try:
            new_key = keyfn(key)
        except exception:
            if dropfailedkeys:
                continue
            new_key = key
        value = dictin[key]
        try:
            new_value = valuefn(value)
        except exception:
            if dropfailedvalues:
                continue
            new_value = value
        dictout[new_key] = new_value
    return dictout