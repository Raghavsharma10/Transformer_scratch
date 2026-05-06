def proxy(name, default = None):
    """
    Create a proxy module. A proxy module has a default implementation, but can be redirected to other
    implementations with configurations. Other modules can depend on proxy modules.
    """
    proxymodule = _ProxyMetaClass(name, (_ProxyModule,), {'_default': default})
    proxymodule.__module__ = sys._getframe(1).f_globals.get('__name__')
    return proxymodule