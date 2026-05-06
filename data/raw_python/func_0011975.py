def _no_exp(number):
    r"""
    Convert a number to a string without using scientific notation.

    :param number: Number to convert
    :type  number: integer or float

    :rtype: string

    :raises: RuntimeError (Argument \`number\` is not valid)
    """
    if isinstance(number, bool) or (not isinstance(number, (int, float))):
        raise RuntimeError("Argument `number` is not valid")
    mant, exp = _to_scientific_tuple(number)
    if not exp:
        return str(number)
    floating_mant = "." in mant
    mant = mant.replace(".", "")
    if exp < 0:
        return "0." + "0" * (-exp - 1) + mant
    if not floating_mant:
        return mant + "0" * exp + (".0" if isinstance(number, float) else "")
    lfpart = len(mant) - 1
    if lfpart < exp:
        return (mant + "0" * (exp - lfpart)).rstrip(".")
    return mant