def round_to_nearest(number, nearest=1):
    """Round 'number' to the nearest multiple of 'nearest'.

    Parameters
    ----------
    number
        A real number to round.
    nearest
        Number to round to closes multiple of.

    Returns
    -------
    rounded
        A rounded number.


    Examples
    -------
    >>> round_to_nearest(6.8, nearest = 2.5)
    7.5
    """
    result = nearest * round(number / nearest)
    if result % 1 == 0:
        return int(result)

    if nearest % 1 == 0:
        return round(result)
    if nearest % 0.1 == 0:
        return round(result, 1)
    if nearest % 0.01 == 0:
        return round(result, 2)
    return result