def _translate_args(args, desired_locale=None):
    """Translates all the translatable elements of the given arguments object.

    This method is used for translating the translatable values in method
    arguments which include values of tuples or dictionaries.
    If the object is not a tuple or a dictionary the object itself is
    translated if it is translatable.

    If the locale is None the object is translated to the system locale.

    :param args: the args to translate
    :param desired_locale: the locale to translate the args to, if None the
                           default system locale will be used
    :returns: a new args object with the translated contents of the original
    """
    if isinstance(args, tuple):
        return tuple(translate(v, desired_locale) for v in args)
    if isinstance(args, dict):
        translated_dict = {}
        for (k, v) in six.iteritems(args):
            translated_v = translate(v, desired_locale)
            translated_dict[k] = translated_v
        return translated_dict
    return translate(args, desired_locale)