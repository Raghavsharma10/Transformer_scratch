def usage(text):
    """
    Decorator used to specify a usage string for the console script
    help message.

    :param text: The text to use for the usage.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.usage = text
        return func
    return decorator