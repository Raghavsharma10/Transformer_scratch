def get_task_parser(task):
    """
    Construct an ArgumentParser for the given task.

    This function returns a tuple (parser, proxy_args).
    If task accepts varargs only, proxy_args is True.
    If task accepts only positional and explicit keyword args,
    proxy args is False.
    """
    args, varargs, keywords, defaults = inspect.getargspec(task)
    defaults = defaults or []
    parser = argparse.ArgumentParser(
        prog='ape ' + task.__name__,
        add_help=False,
        description=task.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    posargslen = len(args) - len(defaults)
    if varargs is None and keywords is None:
        for idx, arg in enumerate(args):
            if idx < posargslen:
                parser.add_argument(arg)
            else:
                default = defaults[idx - posargslen]
                parser.add_argument('--' + arg, default=default)
        return parser, False
    elif not args and varargs and not keywords and not defaults:
        return parser, True
    else:
        raise InvalidTask(ERRMSG_UNSUPPORTED_SIG % task.__name__)