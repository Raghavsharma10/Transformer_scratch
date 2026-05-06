def import_qualified(name):
    '''
    Imports a fully-qualified name from a module:

        cls = import_qualified('homepage.views.index.MyForm')

    Raises an ImportError if it can't be ipmorted.
    '''
    parts = name.rsplit('.', 1)
    if len(parts) != 2:
        raise ImportError('Invalid fully-qualified name: {}'.format(name))
    try:
        return getattr(import_module(parts[0]), parts[1])
    except AttributeError:
        raise ImportError('{} not found in module {}'.format(parts[1], parts[0]))