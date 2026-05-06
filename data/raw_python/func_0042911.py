def reverse_collections(self, key, value):
    """Reverse colections field to custom MARC tag 980."""
    return {
        'a': value.get('primary'),
        'b': value.get('secondary'),
        'c': value.get('deleted'),
    }