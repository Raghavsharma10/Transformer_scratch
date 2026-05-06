def i18n_install(lc=None):
    """
    Install internationalization support for the clients using the specified locale.
    If there is no support for the locale, the default locale will be used.
    As last resort, a null translator will be installed.
    :param lc: locale to install. If None, the system default locale will be used.
    """
    log.debug('i18n_install( {lc} ) called.'.format(lc=lc))
    if lc is None:
        lc = i18n_system_locale()
    if lc is None:
        log.debug('i18n_install(): installing NullTranslations')
        translator = gettext.NullTranslations()
    else:
        child_locales = i18n_support_locale(lc)  # Call i18n_support_locale to log the supported locales

        log.debug('i18n_install(): installing gettext.translation(domain={domain}, localedir={localedir}, '
                  'languages={languages}, fallback={fallback})'.format(domain=project.PROJECT_TITLE.lower(),
                                                                       localedir=i18n_get_path(),
                                                                       languages=child_locales,
                                                                       fallback=True))
        translator = gettext.translation(
            domain=project.PROJECT_TITLE.lower(), localedir=str(i18n_get_path()),
            languages=child_locales, fallback=True)
    translator.install(names=['ngettext'])