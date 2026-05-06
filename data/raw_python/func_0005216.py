def import_from(name):
    """ Imports a module, class or method from string and unwraps it
        if wrapped by functools

        @name: (#str) name of the python object

        -> imported object
    """
    obj = name
    if isinstance(name, str) and len(name):
        try:
            obj = locate(name)
            assert obj is not None
        except (AttributeError, TypeError, AssertionError, ErrorDuringImport):
            try:
                name = name.split(".")
                attr = name[-1]
                name = ".".join(name[:-1])
                mod = importlib.import_module(name)
                obj = getattr(mod, attr)
            except (SyntaxError, AttributeError, ImportError, ValueError):
                try:
                    name = name.split(".")
                    attr_sup = name[-1]
                    name = ".".join(name[:-1])
                    mod = importlib.import_module(name)
                    obj = getattr(getattr(mod, attr_sup), attr)
                except:
                    # We give up.
                    pass

    obj = unwrap_obj(obj)
    return obj