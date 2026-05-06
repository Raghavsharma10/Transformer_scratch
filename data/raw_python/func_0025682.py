def defvar(varname):
    """Returns true if CL variable is defined."""

    if 'pyraf' in sys.modules:
        #ONLY if pyraf is already loaded, import iraf into the namespace
        from pyraf import iraf
    else:
        # else set iraf to None so it knows to not use iraf's environment
        iraf = None

    if iraf:
        _irafdef = iraf.envget(varname)
    else:
        _irafdef = 0
    return varname in _varDict or varname in os.environ or _irafdef