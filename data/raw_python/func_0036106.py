def is_zsettable(s):
    """quick check that all values in a dict are reals"""
    return all(map(lambda x: isinstance(x, (int, float, long)), s.values()))