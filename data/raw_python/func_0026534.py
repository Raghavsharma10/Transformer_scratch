def all_languages():
    """Compile a list of all available language translations"""

    rv = []

    for lang in os.listdir(localedir):
        base = lang.split('_')[0].split('.')[0].split('@')[0]
        if 2 <= len(base) <= 3 and all(c.islower() for c in base):
            if base != 'all':
                rv.append(lang)
    rv.sort()
    rv.append('en')
    l10n_log('Registered languages:', rv, lvl=verbose)

    return rv