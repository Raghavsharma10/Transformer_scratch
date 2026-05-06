def translate(obj, desired_locale=None):
    """Gets the translated unicode representation of the given object.

    If the object is not translatable it is returned as-is.
    If the locale is None the object is translated to the system locale.

    :param obj: the object to translate
    :param desired_locale: the locale to translate the message to, if None the
                           default system locale will be used
    :returns: the translated object in unicode, or the original object if
              it could not be translated
    """
    message = obj
    if not isinstance(message, Message):
        # If the object to translate is not already translatable,
        # let's first get its unicode representation
        message = six.text_type(obj)
    if isinstance(message, Message):
        # Even after unicoding() we still need to check if we are
        # running with translatable unicode before translating
        return message.translate(desired_locale)
    return obj