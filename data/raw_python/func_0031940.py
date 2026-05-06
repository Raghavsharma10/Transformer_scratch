def get_environ(keys):
    """
    Get environment variables from :data:`os.environ`.

    :type keys: [str]
    :rtype: dict

    Some additional features.

    * If 'HOST' is not in :data:`os.environ`, this function
      automatically fetch it using :meth:`platform.node`.
    * If 'TTY' is not in :data:`os.environ`, this function
      automatically fetch it using :meth:`os.ttyname`.
    * Set 'RASH_SPENV_TERMINAL' if needed.

    """
    items = ((k, os.environ.get(k)) for k in keys)
    subenv = dict((k, v) for (k, v) in items if v is not None)
    needset = lambda k: k in keys and not subenv.get(k)

    def setifnonempty(key, value):
        if value:
            subenv[key] = value

    if needset('HOST'):
        import platform
        subenv['HOST'] = platform.node()
    if needset('TTY'):
        setifnonempty('TTY', get_tty())
    if needset('RASH_SPENV_TERMINAL'):
        from .utils.termdetection import detect_terminal
        setifnonempty('RASH_SPENV_TERMINAL', detect_terminal())
    return subenv