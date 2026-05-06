def _module_iterator(root, recursive=True):
    """Iterate over modules."""
    yield root

    stack = collections.deque((root,))
    while stack:
        package = stack.popleft()
        # The '__path__' attribute of a package might return a list of paths if
        # the package is referenced as a namespace.
        paths = getattr(package, '__path__', [])
        for path in paths:
            modules = pkgutil.iter_modules([path])
            for finder, name, is_package in modules:
                module_name = '%s.%s' % (package.__name__, name)
                module = sys.modules.get(module_name, None)
                if module is None:
                    # Import the module through the finder to support package
                    # namespaces.
                    module = _load_module(finder, module_name)

                if is_package:
                    if recursive:
                        stack.append(module)
                        yield module
                else:
                    yield module