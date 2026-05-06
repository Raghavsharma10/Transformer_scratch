def safe_int(value):
    """
    Tries to convert a value to int; returns 0 if conversion failed
    """
    try:
        result = int(value)
        if result < 0:
            raise NegativeDurationError(
                'Negative values in duration strings are not allowed!'
            )
    except NegativeDurationError as exc:
        raise exc
    except (TypeError, ValueError):
        result = 0
    return result