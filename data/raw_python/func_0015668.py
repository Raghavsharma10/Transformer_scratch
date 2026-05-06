def require_foreign(namespace, symbol=None):
    """Raises ImportError if the specified foreign module isn't supported or
    the needed dependencies aren't installed.

    e.g.: check_foreign('cairo', 'Context')
    """

    
    try:
        if symbol is None:
            get_foreign_module(namespace)
        else:
            get_foreign_struct(namespace, symbol)
    except ForeignError as e:
        raise ImportError(e)