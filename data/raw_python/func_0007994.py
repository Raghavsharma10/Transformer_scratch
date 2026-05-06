def getlang_by_alpha2(code):
    """
    Lookup a Language object for language code `code` based on these strategies:
      - Special case rules for Hebrew and Chinese Hans/Hant scripts
      - Using `alpha_2` lookup in `pycountry.languages` followed by lookup for a
        language with the same `name` in the internal representaion
    Returns `None` if no matching language is found.
    """
    # Handle special cases for language codes returned by YouTube API
    if code == 'iw':   # handle old Hebrew code 'iw' and return modern code 'he'
        return getlang('he')
    elif 'zh-Hans' in code:
        return getlang('zh-CN')   # use code `zh-CN` for all simplified Chinese
    elif 'zh-Hant' in code or re.match('zh(.*)?-HK', code):
        return getlang('zh-TW')   # use code `zh-TW` for all traditional Chinese

    # extract prefix only if specified with subcode: e.g. zh-Hans --> zh
    first_part = code.split('-')[0]

    # See if pycountry can find this language
    try:
        pyc_lang = pycountry.languages.get(alpha_2=first_part)
        if pyc_lang:
            if hasattr(pyc_lang, 'inverted_name'):
                lang_name = pyc_lang.inverted_name
            else:
                lang_name = pyc_lang.name
            return getlang_by_name(lang_name)
        else:
            return None
    except KeyError:
        return None