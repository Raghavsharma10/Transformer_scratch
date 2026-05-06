def argument_group(group, **kwargs):
    """
    Decorator used to specify an argument group.  Keyword arguments
    have the same meaning as those given to
    ``argparse.ArgumentParser.add_argument_group()``.

    Arguments may be placed in a given argument group by passing the
    ``group`` keyword argument to @argument().

    :param group: The name of the argument group.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor._add_group(group, 'group', kwargs)
        return func
    return decorator