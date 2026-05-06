def execute_scriptfunction() -> None:
    """Execute a HydPy script function.

    Function |execute_scriptfunction| is indirectly applied and
    explained in the documentation on module |hyd|.
    """
    try:
        args_given = []
        kwargs_given = {}
        for arg in sys.argv[1:]:
            if len(arg) < 3:
                args_given.append(arg)
            else:
                try:
                    key, value = parse_argument(arg)
                    kwargs_given[key] = value
                except ValueError:
                    args_given.append(arg)
        logfilepath = prepare_logfile(kwargs_given.pop('logfile', 'stdout'))
        logstyle = kwargs_given.pop('logstyle', 'plain')
        try:
            funcname = str(args_given.pop(0))
        except IndexError:
            raise ValueError(
                'The first positional argument defining the function '
                'to be called is missing.')
        try:
            func = hydpy.pub.scriptfunctions[funcname]
        except KeyError:
            available_funcs = objecttools.enumeration(
                sorted(hydpy.pub.scriptfunctions.keys()))
            raise ValueError(
                f'There is no `{funcname}` function callable by `hyd.py`.  '
                f'Choose one of the following instead: {available_funcs}.')
        args_required = inspect.getfullargspec(func).args
        nmb_args_required = len(args_required)
        nmb_args_given = len(args_given)
        if nmb_args_given != nmb_args_required:
            enum_args_given = ''
            if nmb_args_given:
                enum_args_given = (
                    f' ({objecttools.enumeration(args_given)})')
            enum_args_required = ''
            if nmb_args_required:
                enum_args_required = (
                    f' ({objecttools.enumeration(args_required)})')
            raise ValueError(
                f'Function `{funcname}` requires `{nmb_args_required:d}` '
                f'positional arguments{enum_args_required}, but '
                f'`{nmb_args_given:d}` are given{enum_args_given}.')
        with _activate_logfile(logfilepath, logstyle, 'info', 'warning'):
            func(*args_given, **kwargs_given)
    except BaseException as exc:
        if logstyle not in LogFileInterface.style2infotype2string:
            logstyle = 'plain'
        with _activate_logfile(logfilepath, logstyle, 'exception', 'exception'):
            arguments = ', '.join(sys.argv)
            print(f'Invoking hyd.py with arguments `{arguments}` '
                  f'resulted in the following error:\n{str(exc)}\n\n'
                  f'See the following stack traceback for debugging:\n',
                  file=sys.stderr)
            traceback.print_tb(sys.exc_info()[2])