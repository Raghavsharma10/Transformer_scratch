def debug(s, *args):
    """debug(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    # Get the path name and line number of the function which called us.
    previous_frame = inspect.currentframe().f_back
    try:
        pathname, lineno, _, _, _ = inspect.getframeinfo(previous_frame)
        # if path is in cwd, simplify it
        cwd = os.path.abspath(os.getcwd())
        pathname = os.path.abspath(pathname)
        if os.path.commonprefix([cwd, pathname]) == cwd:
            pathname = os.path.relpath(pathname, cwd)
    except Exception:  # pylint: disable=broad-except
        pathname = '<UNKNOWN-FILE>.py'
        lineno = 0
    if _FORMATTER: # log could have not been initialized.
        _FORMATTER.pathname = pathname
        _FORMATTER.lineno = lineno
    logger = logging.getLogger(__package__)
    logger.debug(s.format(*args))