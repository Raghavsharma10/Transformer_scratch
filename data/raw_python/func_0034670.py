def to_iso639_2(key, type='B'):
    """Find ISO 639-2 code for language specified by key.

    :param type: "B" - bibliographical (default), "T" - terminological

    >>> to_iso639_2("German")
    u'ger'
    >>> to_iso639_2("German", "T")
    u'deu'
    """
    if type not in ('B', 'T'):
        raise ValueError('Type must be either "B" or "T".')
    item = find(whatever=key)
    if not item:
        raise NonExistentLanguageError('Language does not exist.')
    if type == 'T' and item[u'iso639_2_t']:
        return item[u'iso639_2_t']
    return item[u'iso639_2_b']