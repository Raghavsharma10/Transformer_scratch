def collections(record, key, value):
    """Parse custom MARC tag 980."""
    return {
        'primary': value.get('a'),
        'secondary': value.get('b'),
        'deleted': value.get('c'),
    }