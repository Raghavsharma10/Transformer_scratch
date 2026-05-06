def decorator(self, func):
        """ Wrapper function to decorate a function """
        if inspect.isfunction(func):
            func._methodview = self
        elif inspect.ismethod(func):
            func.__func__._methodview = self
        else:
            raise AssertionError('Can only decorate function and methods, {} given'.format(func))
        return func