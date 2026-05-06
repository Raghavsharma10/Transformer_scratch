def i18n_get_supported_locales():
    """
    List all locales that have internationalization data for this program
    :return: List of locales
    """
    locale_path = i18n_get_path()
    log.debug('Scanning translation files .mo in locale path: {}'.format(locale_path))
    langs = []
    mo_file = '{project}.mo'.format(project=project.PROJECT_TITLE.lower())
    for lc in locale_path.iterdir():
        lc_mo_path = lc / 'LC_MESSAGES' / mo_file
        if lc_mo_path.exists():
            langs.append(lc.name)
    log.debug('Detected: {langs}'.format(langs=langs))
    return langs