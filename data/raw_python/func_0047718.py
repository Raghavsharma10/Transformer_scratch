def is_authenticated_with_proxy(proxy):
    """Given a Proxy, checks whether a user is authenticated"""
    if proxy is None:
        return False
    elif proxy.has_authentication():
        return proxy.get_authentication().is_valid()
    else:
        return False