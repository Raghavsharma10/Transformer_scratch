def _encode(self, obj, context):
        """Encodes a class to a lower-level object using the class' own
        to_construct function.
        If no such function is defined, returns the object unchanged.
        """
        func = getattr(obj, 'to_construct', None)
        if callable(func):
            return func(context)
        else:
            return obj