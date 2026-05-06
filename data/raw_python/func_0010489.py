def i18n_locale_fallbacks_calculate(lc):
    """
    Calculate all child locales from a locale.
    e.g. for locale="pt_BR.us-ascii", returns ["pt_BR.us-ascii", "pt_BR.us", "pt_BR", "pt"]
    :param lc: locale for which the child locales are needed
    :return: all child locales (including the parameter lc)
    """
    log.debug('i18n_locale_fallbacks_calculate( locale="{locale}" ) called'.format(locale=lc))
    locales = []
    lc_original = lc
    while lc:
        locales.append(lc)
        rindex = max([lc.rfind(separator) for separator in ['@', '_', '-', '.']])
        if rindex == -1:
            break
        lc = lc[:rindex]
    log.debug('i18n_locale_fallbacks_calculate( lc="{lc}" ) = {locales}'.format(lc=lc_original, locales=locales))
    return locales