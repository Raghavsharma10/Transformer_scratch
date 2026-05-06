def isset(alias_name):
    """Return a boolean if the docker link is set or not and is a valid looking docker link value.

    Args:
        alias_name: The link alias name
    """
    warnings.warn('Will be removed in v1.0', DeprecationWarning, stacklevel=2)
    raw_value = read(alias_name, allow_none=True)
    if raw_value:
        if re.compile(r'.+://.+:\d+').match(raw_value):
            return True
        else:
            warnings.warn('"{0}_PORT={1}" does not look like a docker link.'.format(alias_name, raw_value), stacklevel=2)
            return False

    return False