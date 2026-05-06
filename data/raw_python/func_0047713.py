def remove_null_proxy_kwarg(func):
    """decorator, to remove a 'proxy' keyword argument. For wrapping certain Manager methods"""
    def wrapper(*args, **kwargs):
        if 'proxy' in kwargs:
            # if kwargs['proxy'] is None:
            del kwargs['proxy']
            # else:
            #    raise InvalidArgument('Manager sessions cannot be called with Proxies. Use ProxyManager instead')
        return func(*args, **kwargs)
    return wrapper