def getlang_by_name(name):
    """
    Try to lookup a Language object by name, e.g. 'English', in internal language list.
    Returns None if lookup by language name fails in resources/languagelookup.json.
    """
    direct_match = _iget(name, _LANGUAGE_NAME_LOOKUP)
    if direct_match:
        return direct_match
    else:
        simple_name = name.split(',')[0]                 # take part before comma
        simple_name = simple_name.split('(')[0].strip()  # and before any bracket
        return _LANGUAGE_NAME_LOOKUP.get(simple_name, None)