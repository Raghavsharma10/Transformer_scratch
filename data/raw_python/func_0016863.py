def _proxy(method):
    """
    Decorator returning a method that proxies a data source.
    """
    @functools.wraps(method)
    def memoizer(self, context):
        return method(context)

    return memoizer