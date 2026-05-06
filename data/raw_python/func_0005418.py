def fields(cls):
        """Get the names of all writable properties that reflect
        metadata tags (i.e., those that are instances of
        :class:`MediaField`).
        """
        for property, descriptor in cls.__dict__.items():
            if isinstance(descriptor, MediaField):
                if isinstance(property, bytes):
                    # On Python 2, class field names are bytes. This method
                    # produces text strings.
                    yield property.decode('utf8', 'ignore')
                else:
                    yield property