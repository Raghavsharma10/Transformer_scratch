def from_string(cls, s, space):
        """Produce a TopNumber by hashing a string."""

        import hashlib

        hs = hashlib.sha1(s).hexdigest()

        return cls.from_hex(hs, space)