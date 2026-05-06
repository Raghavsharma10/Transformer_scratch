def _multilingual(function, *args, **kwargs):
    """ Returns the value from the function with the given name in the given language module.
        By default, language="en".
    """
    return getattr(_module(kwargs.pop("language", "en")), function)(*args, **kwargs)