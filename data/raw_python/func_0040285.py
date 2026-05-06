def continues(method):
        '''Method decorator signifying that the visitor should not visit the
        current node's children once this method has been invoked.
        '''
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            yield method(self, *args, **kwargs)
            raise self.Continue()
        return wrapped