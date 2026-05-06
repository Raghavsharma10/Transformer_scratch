def _import_module(name, path=None):
    """
    Args:
        name(str):
            * Full name of object
            * name can also be an EntryPoint object, name and path will be determined dynamically
        path(str): Module directory

    Returns:
        object: module object or advertised object for EntryPoint

    Loads a module using importlib catching exceptions
    If path is given, the traceback will be formatted to give more friendly and direct information
    """

    # If name is an entry point, try to parse it
    epoint = None
    if isinstance(name, EntryPoint):
        epoint = name
        name = epoint.module_name

    if path is None:
        try:
            loader = pkgutil.get_loader(name)
        except ImportError:
            pass
        else:
            if loader:
                path = os.path.dirname(loader.get_filename(name))

    LOGGER.debug('Attempting to load module %s from %s', name, path)
    try:
        if epoint:
            mod = epoint.load()
        else:
            mod = importlib.import_module(name)

    except Exception as e:  # pylint: disable=broad-except

        etype = e.__class__
        tback = getattr(e, '__traceback__', sys.exc_info()[2])

        # Create traceback starting at module for friendly output
        start = 0
        here = 0
        tb_list = traceback.extract_tb(tback)

        if path:
            for idx, entry in enumerate(tb_list):
                # Find index for traceback starting with module we tried to load
                if os.path.dirname(entry[0]) == path:
                    start = idx
                    break
                # Find index for traceback starting with this file
                elif os.path.splitext(entry[0])[0] == os.path.splitext(__file__)[0]:
                    here = idx

        if start == 0 and isinstance(e, SyntaxError):
            limit = 0
        else:
            limit = 0 - len(tb_list) + max(start, here)

        # pylint: disable=wrong-spelling-in-comment
        # friendly = ''.join(traceback.format_exception(etype, e, tback, limit))
        friendly = ''.join(format_exception(etype, e, tback, limit))

        # Format exception
        msg = 'Error while importing candidate plugin module %s from %s' % (name, path)
        exception = PluginImportError('%s: %s' % (msg, repr(e)), friendly=friendly)

        raise_with_traceback(exception, tback)

    return mod