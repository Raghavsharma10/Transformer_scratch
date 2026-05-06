def _decode(self, obj, context):
        """Initialises a new Python class from a construct using the mapping
        passed to the adapter.
        """
        cls = self._get_class(obj.classID)
        return cls.from_construct(obj, context)