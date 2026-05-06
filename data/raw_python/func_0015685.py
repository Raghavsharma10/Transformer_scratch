def _generate_constructor(cls, names):
        """Get a hopefully cache constructor"""

        cache = cls._constructors
        if names in cache:
            return cache[names]
        elif len(cache) > 3:
            cache.clear()

        func = generate_constructor(cls, names)
        cache[names] = func
        return func