def list_modules(root_package = 'vlcp'):
    '''
    Walk through all the sub modules, find subclasses of vlcp.server.module.Module,
    list their apis through apidefs
    '''
    pkg = __import__(root_package, fromlist=['_'])
    module_dict = OrderedDict()
    _server = Server()
    for imp, module, _ in walk_packages(pkg.__path__, root_package + '.'):
        m = __import__(module, fromlist = ['_'])
        for name, v in vars(m).items():
            if v is not None and isinstance(v, type) and issubclass(v, Module) \
                    and v is not Module \
                    and not isinstance(v, _ProxyModule) \
                    and hasattr(v, '__dict__') and 'configkey' in v.__dict__ \
                    and v.__module__ == module:
                module_name = v.__name__.lower()
                if module_name not in module_dict:
                    _inst = v(_server)
                    module_info = OrderedDict((('class', v.__module__ + '.' + v.__name__),
                                                ('dependencies', [d.__name__.lower()
                                                                  for d in v.depends]),
                                                ('classdescription', getdoc(v)),
                                                ('apis', [])))
                    if hasattr(_inst, 'apiHandler'):
                        apidefs = _inst.apiHandler.apidefs
                        module_info['apis'] = [(d[0], d[3])
                                               for d in apidefs
                                               if len(d) > 3 and \
                                               not d[0].startswith('public/')]
                    module_dict[module_name] = module_info
    return module_dict