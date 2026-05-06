def expand_factor_conditions(s, env):
    """If env matches the expanded factor then return value else return ''.

    Example
    -------
    >>> s = 'py{33,34}: docformatter'
    >>> expand_factor_conditions(s, Env(name="py34", ...))
    "docformatter"
    >>> expand_factor_conditions(s, Env(name="py26", ...))
    ""

    """
    try:
        factor, value = re.split(r'\s*\:\s*', s)
    except ValueError:
        return s

    if matches_factor_conditions(factor, env):
        return value
    else:
        return ''