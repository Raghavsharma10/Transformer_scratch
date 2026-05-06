def per(arga, argb, prec=10):
    r"""
    Calculate percentage difference between numbers.

    If only two numbers are given, the percentage difference between them is
    computed. If two sequences of numbers are given (either two lists of
    numbers or Numpy vectors), the element-wise percentage difference is
    computed. If any of the numbers in the arguments is zero the value returned
    is the maximum floating-point number supported by the Python interpreter.

    :param arga: First number, list of numbers or Numpy vector
    :type  arga: float, integer, list of floats or integers, or Numpy vector
                 of floats or integers

    :param argb: Second number, list of numbers or or Numpy vector
    :type  argb: float, integer, list of floats or integers, or Numpy vector
                 of floats or integers

    :param prec: Maximum length of the fractional part of the result
    :type  prec: integer

    :rtype: Float, list of floats or Numpy vector, depending on the arguments
     type

    :raises:
     * RuntimeError (Argument \`arga\` is not valid)

     * RuntimeError (Argument \`argb\` is not valid)

     * RuntimeError (Argument \`prec\` is not valid)

     * TypeError (Arguments are not of the same type)
    """
    # pylint: disable=C0103,C0200,E1101,R0204
    if not isinstance(prec, int):
        raise RuntimeError("Argument `prec` is not valid")
    a_type = 1 * _isreal(arga) + 2 * (isiterable(arga) and not isinstance(arga, str))
    b_type = 1 * _isreal(argb) + 2 * (isiterable(argb) and not isinstance(argb, str))
    if not a_type:
        raise RuntimeError("Argument `arga` is not valid")
    if not b_type:
        raise RuntimeError("Argument `argb` is not valid")
    if a_type != b_type:
        raise TypeError("Arguments are not of the same type")
    if a_type == 1:
        arga, argb = float(arga), float(argb)
        num_min, num_max = min(arga, argb), max(arga, argb)
        return (
            0
            if _isclose(arga, argb)
            else (
                sys.float_info.max
                if _isclose(num_min, 0.0)
                else round((num_max / num_min) - 1, prec)
            )
        )
    # Contortions to handle lists and Numpy arrays without explicitly
    # having to import numpy
    ret = copy.copy(arga)
    for num, (x, y) in enumerate(zip(arga, argb)):
        if not _isreal(x):
            raise RuntimeError("Argument `arga` is not valid")
        if not _isreal(y):
            raise RuntimeError("Argument `argb` is not valid")
        x, y = float(x), float(y)
        ret[num] = (
            0
            if _isclose(x, y)
            else (
                sys.float_info.max
                if _isclose(x, 0.0) or _isclose(y, 0)
                else (round((max(x, y) / min(x, y)) - 1, prec))
            )
        )
    return ret