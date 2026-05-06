def _setup_default_handler(progname, fmt=None, datefmt=None, **_):
    """Create a Stream handler (default handler).

    Args:
        progname (str): Name of program.
        fmt (:obj:`str`, optional): Desired log format if different than
            the default; uses the same formatting string options
            supported in the stdlib's `logging` module.
        datefmt (:obj:`str`, optional): Desired date format if different
            than the default; uses the same formatting string options
            supported in the stdlib's `logging` module.

    Returns:
        (obj): Instance of `logging.StreamHandler`
    """
    handler = logging.StreamHandler()

    if not fmt:
        # ex: 2017-08-26T14:47:44.968+00:00 <progname> (<PID>) INFO: <msg>
        fmt_prefix = '%(asctime)s.%(msecs)03dZ '
        fmt_suffix = ' (%(process)d) %(levelname)s: ' + '%(message)s'
        fmt = fmt_prefix + progname + fmt_suffix
    if not datefmt:
        datefmt = '%Y-%m-%dT%H:%M:%S'

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler