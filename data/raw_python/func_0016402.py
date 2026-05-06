def clone(proxy, persistent=True):
    """factory function for cloning a proxy object"""

    if not isinstance(proxy, _Proxy):
        raise TypeError('argument is not a Proxy object')

    if persistent:
        pclass = _PersistentProxy
    else:
        pclass = _Proxy

    return pclass(proxy._family, proxy._sockaddr,
                  proxy.flags & ~FLG_PERSISTENCE, proxy.verbose, proxy.errmess)