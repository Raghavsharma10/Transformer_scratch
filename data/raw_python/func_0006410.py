def round_to_multiple(number, multiple):
    '''Rounding up to the nearest multiple of any positive integer

    Parameters
    ----------
    number : int, float
        Input number.
    multiple : int
        Round up to multiple of multiple. Will be converted to int. Must not be equal zero.
    Returns
    -------
    ceil_mod_number : int
        Rounded up number.

    Example
    -------
    round_to_multiple(maximum, math.floor(math.log10(maximum)))
    '''
    multiple = int(multiple)
    if multiple == 0:
        multiple = 1
    ceil_mod_number = number - number % (-multiple)
    return int(ceil_mod_number)