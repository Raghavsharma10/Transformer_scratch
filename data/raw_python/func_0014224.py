def is_decorated(cls, f):
        '''Returns True if the given function is decorated with @view_function'''
        real_func = inspect.unwrap(f)
        return real_func in cls.DECORATED_FUNCTIONS