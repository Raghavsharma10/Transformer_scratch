def getlang_by_native_name(native_name):
    """
    Try to lookup a Language object by native_name, e.g. 'English', in internal language list.
    Returns None if lookup by language name fails in resources/languagelookup.json.
    """
    direct_match = _iget(native_name, _LANGUAGE_NATIVE_NAME_LOOKUP)
    if direct_match:
        return direct_match
    else:
        simple_native_name = native_name.split(',')[0]                 # take part before comma
        simple_native_name = simple_native_name.split('(')[0].strip()  # and before any bracket
        return _LANGUAGE_NATIVE_NAME_LOOKUP.get(simple_native_name, None)