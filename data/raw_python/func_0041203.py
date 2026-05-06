def gettext(message):
    """
    Translate the 'message' string. It uses the current thread to find the
    translation object to use. If no current translation is activated, the
    message will be run through the default translation object.
    """
    global _default
    _default = _default or translation(DEFAULT_LANGUAGE)
    translation_object = getattr(_active, 'value', _default)
    result = translation_object.gettext(message)
    return result