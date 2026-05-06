def api(func, container = None, criteria = None):
    '''
    Return an API def for a generic function
    
    :param func: a function or bounded method
    
    :param container: if None, this is used as a synchronous method, the return value of the method
                      is used for the return value. If not None, this is used as an asynchronous method,
                      the return value should be a generator, and it is executed in `container` as a routine.
                      The return value should be set to `container.retvalue`.
    
    :param criteria: An extra function used to test whether this function should process the API. This allows
                     multiple API definitions to use the same API method name.
    '''
    return (func.__name__.lower(), functools.update_wrapper(lambda n,p: func(**p), func), container,
            create_discover_info(func), criteria)