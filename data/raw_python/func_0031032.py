def listify(obj, ignore=(list, tuple, type(None))):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, ignore) else [obj]