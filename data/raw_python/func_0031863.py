def parse_duration(string):
    """
    Parse human readable duration.

    >>> parse_duration('1m')
    60
    >>> parse_duration('7 days') == 7 * 24 * 60 * 60
    True

    """
    if string.isdigit():
        return int(string)
    try:
        return float(string)
    except ValueError:
        pass
    string = string.rstrip()
    for (suf, mult) in DURATION_SUFFIX_MAP.items():
        if string.lower().endswith(suf):
            try:
                return parse_duration(string[:-len(suf)].strip()) * mult
            except TypeError:
                return