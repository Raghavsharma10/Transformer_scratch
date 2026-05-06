def i18n_support_locale(lc_parent):
    """
    Find out whether lc is supported. Returns all child locales (and eventually lc) which do have support.
    :param lc_parent: Locale for which we want to know the child locales that are supported
    :return: list of supported locales
    """
    log.debug('i18n_support_locale( locale="{locale}" ) called'.format(locale=lc_parent))
    lc_childs = i18n_locale_fallbacks_calculate(lc_parent)
    locales = []

    locale_path = i18n_get_path()
    mo_file = '{project}.mo'.format(project=project.PROJECT_TITLE.lower())

    for lc in lc_childs:
        lc_mo_path = locale_path / lc / 'LC_MESSAGES' / mo_file
        log.debug('Locale data "{lc_mo_path}" exists? ...'.format(lc_mo_path=lc_mo_path))
        if lc_mo_path.is_file():
            log.debug('... Yes! "{locale_path}" contains {mo_file}.'.format(locale_path=locale_path, mo_file=mo_file))
            locales.append(lc)
        else:
            log.debug('... No')

    log.debug('i18n_support_locale( lc="{lc}" ) = {locales}'.format(lc=lc_parent, locales=locales))
    return locales