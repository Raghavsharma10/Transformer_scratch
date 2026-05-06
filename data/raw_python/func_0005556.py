def prog(text):
    """
    Decorator used to specify the program name for the console script
    help message.

    :param text: The text to use for the program name.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.prog = text
        return func
    return decorator