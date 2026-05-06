def round(value, decimal=7, digits=None):
    """
    ROUND TO GIVEN NUMBER OF DIGITS, OR GIVEN NUMBER OF DECIMAL PLACES
    decimal - NUMBER OF DIGITS AFTER DECIMAL POINT (NEGATIVE IS VALID)
    digits - NUMBER OF SIGNIFICANT DIGITS (LESS THAN 1 IS INVALID)
    """
    if value == None:
        return None
    else:
        value = float(value)

    if digits != None:
        if digits <= 0:
            if value == 0:
                return int(_round(value, digits))
            try:
                m = pow(10, math_ceil(math_log10(abs(value))))
                return int(_round(value / m, digits) * m)
            except Exception as e:
                from mo_logs import Log

                Log.error("not expected", e)
        else:
            if value == 0:
                return _round(value, digits)
            try:
                m = pow(10, math_ceil(math_log10(abs(value))))
                return _round(value / m, digits) * m
            except Exception as e:
                from mo_logs import Log
                Log.error("not expected", e)
    elif decimal <= 0:
        return int(_round(value, decimal))
    else:
        return _round(value, decimal)