def to_name(key):
    """Find the English name for the language specified by key.

    >>> to_name('br')
    u'Breton'
    >>> to_name('sw')
    u'Swahili'
    """
    item = find(whatever=key)
    if not item:
        raise NonExistentLanguageError('Language does not exist.')
    return item[u'name']