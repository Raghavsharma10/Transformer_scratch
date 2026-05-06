def _timedeltaToSignHrMin(offset):
    """
    Return a (sign, hour, minute) triple for the offset described by timedelta.

    sign is a string, either "+" or "-". In the case of 0 offset, sign is "+".
    """
    minutes = round((offset.days * 3600000000 * 24
                     + offset.seconds * 1000000
                     + offset.microseconds)
                    / 60000000.0)
    if minutes < 0:
        sign = '-'
        minutes = -minutes
    else:
        sign = '+'
    return (sign, minutes // 60, minutes % 60)