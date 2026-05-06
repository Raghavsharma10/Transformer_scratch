def register(cls, origin_type):
        """Decorator for mutation tracker registration.

        The provided `origin_type` is mapped to the decorated class such that
        future calls to `convert()` will convert the object of `origin_type`
        to an instance of the decorated class.
        """
        def decorator(tracked_type):
            """Adds the decorated class to the `_type_mapping` dictionary."""
            cls._type_mapping[origin_type] = tracked_type
            return tracked_type
        return decorator