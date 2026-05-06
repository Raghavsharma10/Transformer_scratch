def InitLog(file_name=None, log_level=logging.DEBUG,
            screen_level=logging.CRITICAL, pdb=False):
    '''
    A little routine to initialize the logging functionality.

    :param str file_name: The name of the file to log to. \
           Default :py:obj:`None` (set internally by :py:mod:`everest`)
    :param int log_level: The file logging level (0-50). Default 10 (debug)
    :param int screen_level: The screen logging level (0-50). \
           Default 50 (critical)

    '''

    # Initialize the logging
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.DEBUG)

    # File handler
    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        fh = logging.FileHandler(file_name)
        fh.setLevel(log_level)
        fh_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(name)s.%(funcName)s()]: %(message)s",
            datefmt="%m/%d/%y %H:%M:%S")
        fh.setFormatter(fh_formatter)
        fh.addFilter(NoPILFilter())
        root.addHandler(fh)

    # Screen handler
    sh = logging.StreamHandler(sys.stdout)
    if pdb:
        sh.setLevel(logging.DEBUG)
    else:
        sh.setLevel(screen_level)
    sh_formatter = logging.Formatter(
        "%(levelname)-5s [%(name)s.%(funcName)s()]: %(message)s")
    sh.setFormatter(sh_formatter)
    sh.addFilter(NoPILFilter())
    root.addHandler(sh)

    # Set exception hook
    if pdb:
        sys.excepthook = ExceptionHookPDB
    else:
        sys.excepthook = ExceptionHook