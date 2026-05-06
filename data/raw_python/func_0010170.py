def default(self, obj):
        """Fired when an unserializable object is hit."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__.copy()
        elif HAS_NUMPY and isinstance(obj, np.ndarray):
            return obj.copy().tolist()
        else:
            raise TypeError(("Object of type {:s} with value of {:s} is not "
                             "JSON serializable").format(type(obj), repr(obj)))