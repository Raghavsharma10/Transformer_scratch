def parse_bool(v, default=None, exceptions: bool=True) -> bool:
    """
    Parses boolean value
    :param v: Input string
    :param default: Default value if exceptions=False
    :param exceptions: Raise exception on error or not
    :return: bool
    """
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in TRUE_VALUES:
        return True
    elif s in FALSE_VALUES:
        return False
    else:
        if exceptions:
            raise ValidationError('Failed to parse boolean from "{}"'.format(v))
        return default