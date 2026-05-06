def return_collection(collection_type):
    """Change method return value from raw API output to collection of models
    """
    def outer_func(func):
        @functools.wraps(func)
        def inner_func(self, *pargs, **kwargs):
            result = func(self, *pargs, **kwargs)
            return list(map(collection_type, result))
        return inner_func
    return outer_func