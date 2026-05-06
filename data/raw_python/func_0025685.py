def unset(*args, **kw):
    """
    Unset IRAF environment variables.

    This is not a standard IRAF task, but it is obviously useful.  It makes the
    resulting variables undefined.  It silently ignores variables that are not
    defined.  It does not change the os environment variables.
    """

    if len(kw) != 0:
        raise SyntaxError("unset requires a list of variable names")

    for arg in args:
        if arg in _varDict:
            del _varDict[arg]