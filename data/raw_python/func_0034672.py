def to_native(key):
    """Find the native name for the language specified by key.

    >>> to_native('br')
    u'brezhoneg'
    >>> to_native('sw')
    u'Kiswahili'
    """
    item = find(whatever=key)
    if not item:
        raise NonExistentLanguageError('Language does not exist.')
    return item[u'native']