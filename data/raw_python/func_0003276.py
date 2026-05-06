def return_self_updater(func):
    '''
    Run func, but still return v. Useful for using knowledge.update with operates like append, extend, etc.
    e.g. return_self(lambda k,v: v.append('newobj'))
    '''
    @functools.wraps(func)
    def decorator(k,v):
        func(k,v)
        return v
    return decorator