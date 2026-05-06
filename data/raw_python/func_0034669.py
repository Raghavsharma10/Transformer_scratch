def to_iso639_1(key):
    """Find ISO 639-1 code for language specified by key.

    >>> to_iso639_1("swe")
    u'sv'
    >>> to_iso639_1("English")
    u'en'
    """
    item = find(whatever=key)
    if not item:
        raise NonExistentLanguageError('Language does not exist.')
    return item[u'iso639_1']