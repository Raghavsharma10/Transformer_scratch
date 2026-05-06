def _validate_desc(self, desc):
        """Validate the predicate description."""
        if desc is None:
            return desc

        if not isinstance(desc, STRING_TYPES):
            raise TypeError(
                "predicate description for Matching must be a string, "
                "got %r" % (type(desc),))

        # Python 2 mandates __repr__ to be an ASCII string,
        # so if Unicode is passed (usually due to unicode_literals),
        # it should be ASCII-encodable.
        if not IS_PY3 and isinstance(desc, unicode):
            try:
                desc = desc.encode('ascii', errors='strict')
            except UnicodeEncodeError:
                raise TypeError("predicate description must be "
                                "an ASCII string in Python 2")

        return desc