def safe_division(dividend, divisor):
    """
    :return:
        nan: invalid arguments
    :rtype: float
    """

    try:
        divisor = float(divisor)
        dividend = float(dividend)
    except (TypeError, ValueError, AssertionError):
        return float("nan")

    try:
        return dividend / divisor
    except (ZeroDivisionError):
        return float("nan")