def release_scheme(self, value):
        """Validate the release scheme."""
        if value not in KNOWN_RELEASE_SCHEMES:
            msg = "Release scheme %r is not supported! (valid options are %s)"
            raise ValueError(msg % (value, concatenate(map(repr, KNOWN_RELEASE_SCHEMES))))
        set_property(self, 'release_scheme', value)