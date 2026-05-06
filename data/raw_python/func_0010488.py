def i18n_system_locale():
    """
    Return the system locale
    :return: the system locale (as a string)
    """
    log.debug('i18n_system_locale() called')
    lc, encoding = locale.getlocale()
    log.debug('locale.getlocale() = (lc="{lc}", encoding="{encoding}).'.format(lc=lc, encoding=encoding))
    if lc is None:
        lc, encoding = locale.getdefaultlocale()
        log.debug('locale.getdefaultlocale() = (lc="{lc}", encoding="{encoding}).'.format(lc=lc, encoding=encoding))
    return lc