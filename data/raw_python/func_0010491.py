def i18n_get_path():
    """
    Get path to the internationalization data.
    :return: path as a string
    """
    local_locale_path = client_get_path() / 'locale'
    if platform.system() == 'Linux':
        if local_locale_path.exists():
            return local_locale_path
        else:
            return Path('/usr/share/locale')
    else:
        return local_locale_path