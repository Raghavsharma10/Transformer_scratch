def formatter_class(klass):
    """
    Decorator used to specify the formatter class for the console
    script.

    :param klass: The formatter class to use.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.formatter_class = klass
        return func
    return decorator