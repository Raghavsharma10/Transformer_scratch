def serialize(self, tag):
        """Return the literal representation of a tag."""
        handler = getattr(self, f'serialize_{tag.serializer}', None)
        if handler is None:
            raise TypeError(f'Can\'t serialize {type(tag)!r} instance')
        return handler(tag)