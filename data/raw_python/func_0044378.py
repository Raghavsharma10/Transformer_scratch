def _factory(importname, base_class_type, path=None, *args, **kargs):
    ''' Load a module of a given base class type

        Parameter
        --------
        importname: string
            Name of the module, etc. converter
        base_class_type: class type
            E.g converter
        path: Absoulte path of the module
            Neede for extensions. If not given module is in online_monitor
            package
        *args, **kargs:
            Arguments to pass to the object init

        Return
        ------
        Object of given base class type
    '''

    def is_base_class(item):
        return isclass(item) and item.__module__ == importname

    if path:
        # Needed to find the module in forked processes; if you know a better
        # way tell me!
        sys.path.append(path)
        # Absolute full path of python module
        absolute_path = os.path.join(path, importname) + '.py'
        module = imp.load_source(importname, absolute_path)
    else:
        module = import_module(importname)

    # Get the defined base class in the loaded module to be name indendend
    clsmembers = getmembers(module, is_base_class)
    if not len(clsmembers):
        raise ValueError('Found no matching class in %s.' % importname)
    else:
        cls = clsmembers[0][1]
    return cls(*args, **kargs)