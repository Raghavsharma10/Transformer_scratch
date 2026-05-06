def _getel(key, value):
    """Returns an element given a key and value."""
    if key in ['HorizontalRule', 'Null']:
        return elt(key, 0)()
    elif key in ['Plain', 'Para', 'BlockQuote', 'BulletList',
                 'DefinitionList', 'HorizontalRule', 'Null']:
        return elt(key, 1)(value)
    return elt(key, len(value))(*value)