def envget(var, default=None):
    """Get value of IRAF or OS environment variable."""

    if 'pyraf' in sys.modules:
        #ONLY if pyraf is already loaded, import iraf into the namespace
        from pyraf import iraf
    else:
        # else set iraf to None so it knows to not use iraf's environment
        iraf = None

    try:
        if iraf:
            return iraf.envget(var)
        else:
            raise KeyError
    except KeyError:
        try:
            return _varDict[var]
        except KeyError:
            try:
                return os.environ[var]
            except KeyError:
                if default is not None:
                    return default
                elif var == 'TERM':
                    # Return a default value for TERM
                    # TERM gets caught as it is found in the default
                    # login.cl file setup by IRAF.
                    print("Using default TERM value for session.")
                    return 'xterm'
                else:
                    raise KeyError("Undefined environment variable `%s'" % var)