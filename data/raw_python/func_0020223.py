def commit_when_no_transaction(f):
    '''Decorator for committing changes when the instance session is
not in a transaction.'''
    def _(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        return self.session.add(self) if self.session is not None else r
    _.__name__ = f.__name__
    _.__doc__ = f.__doc__
    return _