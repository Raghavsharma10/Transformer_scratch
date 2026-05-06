def get_readers(self, obj):
        """ Return the ids of the people who read the message instance. """
        try:
            o = compat_serializer_attr(self, obj)
            return o.readers
        except Exception:
            return []