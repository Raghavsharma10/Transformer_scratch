def integer_key_convert(dictin, dropfailedkeys=False):
    # type: (DictUpperBound, bool) -> Dict
    """Convert keys of dictionary to integers

    Args:
        dictin (DictUpperBound): Input dictionary
        dropfailedkeys (bool): Whether to drop dictionary entries where key conversion fails. Defaults to False.

    Returns:
        Dict: Dictionary with keys converted to integers

    """
    return key_value_convert(dictin, keyfn=int, dropfailedkeys=dropfailedkeys)