def hook(reverse=False,
         align=False,
         strip_path=False,
         enable_on_envvar_only=False,
         on_tty=False,
         conservative=False,
         styles=None,
         tb=None,
         tpe=None,
         value=None):
    """Hook the current excepthook to the backtrace.

    If `align` is True, all parts (line numbers, file names, etc..) will be
    aligned to the left according to the longest entry.

    If `strip_path` is True, only the file name will be shown, not its full
    path.

    If `enable_on_envvar_only` is True, only if the environment variable
    `ENABLE_BACKTRACE` is set, backtrace will be activated.

    If `on_tty` is True, backtrace will be activated only if you're running
    in a readl terminal (i.e. not piped, redirected, etc..).

    If `convervative` is True, the traceback will have more seemingly original
    style (There will be no alignment by default, 'File', 'line' and 'in'
    prefixes and will ignore any styling provided by the user.)

    See https://github.com/nir0s/backtrace/blob/master/README.md for
    information on `styles`.
    """
    if enable_on_envvar_only and 'ENABLE_BACKTRACE' not in os.environ:
        return

    isatty = getattr(sys.stderr, 'isatty', lambda: False)
    if on_tty and not isatty():
        return

    if conservative:
        styles = CONVERVATIVE_STYLES
        align = align or False
    elif styles:
        for k in STYLES.keys():
            styles[k] = styles.get(k, STYLES[k])
    else:
        styles = STYLES

    # For Windows
    colorama.init()

    def backtrace_excepthook(tpe, value, tb=None):
        # Don't know if we're getting traceback or traceback entries.
        # We'll try to parse a traceback object.
        try:
            traceback_entries = traceback.extract_tb(tb)
        except AttributeError:
            traceback_entries = tb
        parser = _Hook(traceback_entries, align, strip_path, conservative)

        tpe = tpe if isinstance(tpe, str) else tpe.__name__
        tb_message = styles['backtrace'].format('Traceback ({0}):'.format(
            'Most recent call ' + ('first' if reverse else 'last'))) + \
            Style.RESET_ALL
        err_message = styles['error'].format(tpe + ': ' + str(value)) + \
            Style.RESET_ALL

        if reverse:
            parser.reverse()

        _flush(tb_message)
        backtrace = parser.generate_backtrace(styles)
        backtrace.insert(0 if reverse else len(backtrace), err_message)
        for entry in backtrace:
            _flush(entry)

    if tb:
        backtrace_excepthook(tpe=tpe, value=value, tb=tb)
    else:
        sys.excepthook = backtrace_excepthook