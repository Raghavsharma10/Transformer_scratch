def split_all(reference, sep):
    """
    Splits a given string at a given separator or list of separators.

    :param reference: The reference to split.
    :param sep: Separator string or list of separator strings.
    :return: A list of split strings
    """
    parts = partition_all(reference, sep)
    return [p for p in parts if p not in sep]