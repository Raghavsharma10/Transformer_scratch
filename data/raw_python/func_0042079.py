def normalise_key(self, key):
        """Make sure key is a valid python attribute"""
        key = key.replace('-', '_')
        if key.startswith("noy_"):
            key = key[4:]
        return key