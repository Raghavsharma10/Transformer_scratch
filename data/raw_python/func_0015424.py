def findsyspy():
    """
    :return: system python executable
    """
    if not in_venv():
        return sys.executable

    python = basename(realpath(sys.executable))
    prefix = None
    if HAS_ORIG_PREFIX_TXT:
        with open(ORIG_PREFIX_TXT) as op:
            prefix = op.read()
    elif HAS_PY_VENV_CFG:
        prefix = getattr(sys, "_home")

    if not prefix:
        return None

    for folder in os.environ['PATH'].split(os.pathsep):
        if folder and \
                normpath(normcase(folder)).startswith(normcase(normpath(prefix))) and \
                isfile(join(folder, python)):
            return join(folder, python)

    # OSX: Homebrew doesn't leave python in the PATH
    if isfile(join(prefix, "bin", python)):
        return join(prefix, "bin", python)