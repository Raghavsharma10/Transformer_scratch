def list_proxy(root_package = 'vlcp'):
    '''
    Walk through all the sub modules, find subclasses of vlcp.server.module._ProxyModule,
    list their default values
    '''
    proxy_dict = OrderedDict()
    pkg = __import__(root_package, fromlist=['_'])
    for imp, module, _ in walk_packages(pkg.__path__, root_package + '.'):
        m = __import__(module, fromlist = ['_'])
        for _, v in vars(m).items():
            if v is not None and isinstance(v, type) and issubclass(v, _ProxyModule) \
                    and v is not _ProxyModule \
                    and v.__module__ == module \
                    and hasattr(v, '_default'):
                name = v.__name__.lower()
                if name not in proxy_dict:
                    proxy_dict[name] = {'defaultmodule': v._default.__name__.lower(),
                                        'class': repr(v._default.__module__ + '.' + v._default.__name__)}
    return proxy_dict