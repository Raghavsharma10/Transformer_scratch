def register(self, bucket, name_or_func, func=None):
        """
        Add a function to the registry by name
        """
        assert bucket in self, 'Bucket %s is unknown' % bucket
        if func is None and hasattr(name_or_func, '__name__'):
            name = name_or_func.__name__
            func = name_or_func
        elif func:
            name = name_or_func
        if name in self[bucket]:
            raise AlreadyRegistered('The function %s is already registered' % name)

        self[bucket][name] = func