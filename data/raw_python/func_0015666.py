def get_foreign_module(namespace):
    """Returns the module or raises ForeignError"""

    if namespace not in _MODULES:
        try:
            module = importlib.import_module("." + namespace, __package__)
        except ImportError:
            module = None
        _MODULES[namespace] = module

    module = _MODULES.get(namespace)
    if module is None:
        raise ForeignError("Foreign %r structs not supported" % namespace)
    return module