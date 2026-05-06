def to_int_list(values):
    """Converts the given list of vlues into a list of integers. If the
    integer conversion fails (e.g. non-numeric strings or None-values), this
    filter will include a 0 instead."""
    results = []
    for v in values:
        try:
            results.append(int(v))
        except (TypeError, ValueError):
            results.append(0)
    return results