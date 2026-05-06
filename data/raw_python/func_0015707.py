def load_overrides(introspection_module):
    """Loads overrides for an introspection module.

    Either returns the same module again in case there are no overrides or a
    proxy module including overrides. Doesn't cache the result.
    """

    namespace = introspection_module.__name__.rsplit(".", 1)[-1]
    module_keys = [prefix + "." + namespace for prefix in const.PREFIX]

    # We use sys.modules so overrides can import from gi.repository
    # but restore everything at the end so this doesn't have any side effects
    for module_key in module_keys:
        has_old = module_key in sys.modules
        old_module = sys.modules.get(module_key)

    # Create a new sub type, so we can separate descriptors like
    # _DeprecatedAttribute for each namespace.
    proxy_type = type(namespace + "ProxyModule", (OverridesProxyModule, ), {})

    proxy = proxy_type(introspection_module)
    for module_key in module_keys:
        sys.modules[module_key] = proxy

    try:
        override_package_name = 'pgi.overrides.' + namespace

        # http://bugs.python.org/issue14710
        try:
            override_loader = get_loader(override_package_name)

        except AttributeError:
            override_loader = None

        # Avoid checking for an ImportError, an override might
        # depend on a missing module thus causing an ImportError
        if override_loader is None:
            return introspection_module

        override_mod = importlib.import_module(override_package_name)

    finally:
        for module_key in module_keys:
            del sys.modules[module_key]
            if has_old:
                sys.modules[module_key] = old_module

    override_all = []
    if hasattr(override_mod, "__all__"):
        override_all = override_mod.__all__

    for var in override_all:
        try:
            item = getattr(override_mod, var)
        except (AttributeError, TypeError):
            # Gedit puts a non-string in __all__, so catch TypeError here
            continue
        # make sure new classes have a proper __module__
        try:
            if item.__module__.split(".")[-1] == namespace:
                item.__module__ = namespace
        except AttributeError:
            pass
        setattr(proxy, var, item)

    # Replace deprecated module level attributes with a descriptor
    # which emits a warning when accessed.
    for attr, replacement in _deprecated_attrs.pop(namespace, []):
        try:
            value = getattr(proxy, attr)
        except AttributeError:
            raise AssertionError(
                "%s was set deprecated but wasn't added to __all__" % attr)
        delattr(proxy, attr)
        deprecated_attr = _DeprecatedAttribute(
            namespace, attr, value, replacement)
        setattr(proxy_type, attr, deprecated_attr)

    return proxy