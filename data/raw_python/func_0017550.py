def exempt(self, obj):
        """
        decorator to mark a view as exempt from htmlmin.
        """
        name = '%s.%s' % (obj.__module__, obj.__name__)

        @wraps(obj)
        def __inner(*a, **k):
            return obj(*a, **k)

        self._exempt_routes.add(name)
        return __inner