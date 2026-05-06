def argument(*args, **kwargs):
    """
    Decorator used to specify an argument taken by the console script.
    Positional and keyword arguments have the same meaning as those
    given to ``argparse.ArgumentParser.add_argument()``.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        group = kwargs.pop('group', None)
        adaptor._add_argument(args, kwargs, group=group)
        return func
    return decorator