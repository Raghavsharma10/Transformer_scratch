def disk_cache(cls, basename, function, *args, method=True, **kwargs):
        """
        Cache the return value in the correct cache directory. Set 'method' to
        false for static methods.
        """
        @utility.disk_cache(basename, cls.directory(), method=method)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper(*args, **kwargs)