def subparsers(**kwargs):
    """
    Decorator used to specify alternate keyword arguments to pass to
    the ``argparse.ArgumentParser.add_subparsers()`` call.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.subkwargs = kwargs
        adaptor.do_subs = True
        return func
    return decorator