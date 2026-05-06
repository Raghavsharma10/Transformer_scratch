def get_object_from_path(path):
    """
    Import's object from given Python path.
    """
    try:
        return sys.IMPORT_CACHE[path]
    except KeyError:
        _path = path.split('.')
        module_path = '.'.join(_path[:-1])
        class_name = _path[-1]
        module = importlib.import_module(module_path)
        sys.IMPORT_CACHE[path] = getattr(module, class_name)
        return sys.IMPORT_CACHE[path]