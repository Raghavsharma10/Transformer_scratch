def venv_bin(name=None):  # pylint: disable=inconsistent-return-statements
    """ Get the directory for virtualenv stubs, or a full executable path
        if C{name} is provided.
    """
    if not hasattr(sys, "real_prefix"):
        easy.error("ERROR: '%s' is not a virtualenv" % (sys.executable,))
        sys.exit(1)

    for bindir in ("bin", "Scripts"):
        bindir = os.path.join(sys.prefix, bindir)
        if os.path.exists(bindir):
            if name:
                bin_ext = os.path.splitext(sys.executable)[1] if sys.platform == 'win32' else ''
                return os.path.join(bindir, name + bin_ext)
            else:
                return bindir

    easy.error("ERROR: Scripts directory not found in '%s'" % (sys.prefix,))
    sys.exit(1)