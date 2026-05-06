def qualified_name(obj):
    '''Returns the fully-qualified name of the given object'''
    if not hasattr(obj, '__module__'):
        obj = obj.__class__
    module = obj.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__qualname__
    return '{}.{}'.format(module, obj.__qualname__)