def is_ecma_regex(regex):
    """Check if given regex is of type ECMA 262 or not.

    :rtype: bool

    """
    parts = regex.split('/')

    if len(parts) == 1:
        return False

    if len(parts) < 3:
        raise ValueError('Given regex isn\'t ECMA regex nor Python regex.')
    parts.pop()
    parts.append('')

    raw_regex = '/'.join(parts)
    if raw_regex.startswith('/') and raw_regex.endswith('/'):
        return True
    return False