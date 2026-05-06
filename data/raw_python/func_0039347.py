def compare(value1, value2, comparison):
    """
    Compare 2 values

    :type value1: object
    :param value1: The first value to compare.

    :type value2: object
    :param value2: The second value to compare.

    :type comparison: string
    :param comparison: The comparison to make. Can be "is", "or", "and".

    :return: If the value is, or, and of another value
    :rtype: boolean
    """
    if not isinstance(comparison, str):
        raise TypeError("Comparison argument must be a string.")
    if comparison == 'is':
        return value1 == value2
    elif comparison == 'or':
        return value1 or value2
    elif comparison == 'and':
        return value1 and value2
    raise ValueError("Invalid comparison operator specified.")