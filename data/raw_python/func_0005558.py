def description(text):
    """
    Decorator used to specify a short description of the console
    script.  This can be used to override the default, which is
    derived from the docstring of the function.

    :param text: The text to use for the description.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor.description = text
        return func
    return decorator