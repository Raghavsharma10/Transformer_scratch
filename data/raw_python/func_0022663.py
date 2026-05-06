def extract_pattern(fmt):
    """Extracts used strings from a %(foo)s pattern."""
    class FakeDict(object):
        def __init__(self):
            self.seen_keys = set()

        def __getitem__(self, key):
            self.seen_keys.add(key)
            return ''

        def keys(self):
            return self.seen_keys

    fake = FakeDict()
    try:
        fmt % fake
    except TypeError:
        # Formatting error
        pass
    return set(fake.keys())