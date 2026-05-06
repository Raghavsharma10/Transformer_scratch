def epilog(text):
    """
    Decorator used to specify an epilog for the console script help
    message.

    :param text: The text to use for the epilog.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.epilog = text
        return func
    return decorator