def from_irafpath(irafpath):
    """Resolve IRAF path like ``jref$`` into actual file path.

    Parameters
    ----------
    irafpath : str
        Path containing IRAF syntax.

    Returns
    -------
    realpath : str
        Actual file path. If input does not follow ``path$filename``
        format, then this is the same as input.

    Raises
    ------
    ValueError
        The required environment variable is undefined.

    """
    s = irafpath.split('$')

    if len(s) != 2:
        return irafpath
    if len(s[0]) == 0:
        return irafpath

    try:
        refdir = os.environ[s[0]]
    except KeyError:
        raise ValueError('{0} environment variable undefined'.format(s[0]))

    return os.path.join(refdir, s[1])