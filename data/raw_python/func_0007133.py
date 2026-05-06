def get_available_languages(domain):
    """Lists the available languages for the given translation domain.

    :param domain: the domain to get languages for
    """
    if domain in _AVAILABLE_LANGUAGES:
        return copy.copy(_AVAILABLE_LANGUAGES[domain])

    localedir = '%s_LOCALEDIR' % domain.upper()
    find = lambda x: gettext.find(domain,
                                  localedir=os.environ.get(localedir),
                                  languages=[x])

    # NOTE(mrodden): en_US should always be available (and first in case
    # order matters) since our in-line message strings are en_US
    language_list = ['en_US']
    # NOTE(luisg): Babel <1.0 used a function called list(), which was
    # renamed to locale_identifiers() in >=1.0, the requirements master list
    # requires >=0.9.6, uncapped, so defensively work with both. We can remove
    # this check when the master list updates to >=1.0, and update all projects
    list_identifiers = (getattr(localedata, 'list', None) or
                        getattr(localedata, 'locale_identifiers'))
    locale_identifiers = list_identifiers()

    for i in locale_identifiers:
        if find(i) is not None:
            language_list.append(i)

    # NOTE(luisg): Babel>=1.0,<1.3 has a bug where some OpenStack supported
    # locales (e.g. 'zh_CN', and 'zh_TW') aren't supported even though they
    # are perfectly legitimate locales:
    #     https://github.com/mitsuhiko/babel/issues/37
    # In Babel 1.3 they fixed the bug and they support these locales, but
    # they are still not explicitly "listed" by locale_identifiers().
    # That is  why we add the locales here explicitly if necessary so that
    # they are listed as supported.
    aliases = {'zh': 'zh_CN',
               'zh_Hant_HK': 'zh_HK',
               'zh_Hant': 'zh_TW',
               'fil': 'tl_PH'}
    for (locale_, alias) in six.iteritems(aliases):
        if locale_ in language_list and alias not in language_list:
            language_list.append(alias)

    _AVAILABLE_LANGUAGES[domain] = language_list
    return copy.copy(language_list)