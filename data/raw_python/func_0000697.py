def hash(self, value):
        """
        Generate a hash of the given iterable.

        This is for use in a cache key.
        """
        if is_iterable(value):
            value = tuple(to_bytestring(v) for v in value)
        return hashlib.md5(six.b(':').join(value)).hexdigest()