def split_path(path_):
    """
    Split the requested path into (locale, path).

    locale will be empty if it isn't found.
    """
    path = path_.lstrip('/')

    # Use partitition instead of split since it always returns 3 parts
    first, _, rest = path.partition('/')

    lang = first.lower()
    if lang in settings.LANGUAGE_URL_MAP:
        return settings.LANGUAGE_URL_MAP[lang], rest
    else:
        supported = find_supported(first)
        if len(supported):
            return supported[0], rest
        else:
            return '', path