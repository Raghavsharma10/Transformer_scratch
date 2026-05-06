def find_spec(name, package=None):
    """Return the spec for the specified module.
    First, sys.modules is checked to see if the module was already imported. If
    so, then sys.modules[name].__spec__ is returned. If that happens to be
    set to None, then ValueError is raised. If the module is not in
    sys.modules, then sys.meta_path is searched for a suitable spec with the
    value of 'path' given to the finders. None is returned if no spec could
    be found.
    If the name is for submodule (contains a dot), the parent module is
    automatically imported.
    The name and package arguments work the same as importlib.import_module().
    In other words, relative module names (with leading dots) work.
    """
    fullname = resolve_name(name, package) if name.startswith('.') else name
    if fullname not in sys.modules:
        parent_name = fullname.rpartition('.')[0]
        if parent_name:
            # Use builtins.__import__() in case someone replaced it.
            parent = __import__(parent_name, fromlist=['__path__'])
            return _find_spec(fullname, parent.__path__)
        else:
            return _find_spec(fullname, None)
    else:
        module = sys.modules[fullname]
        if module is None:
            return None
        try:
            spec = module.__spec__
        except AttributeError:
            six.raise_from(ValueError('{}.__spec__ is not set'.format(name)), None)
        else:
            if spec is None:
                raise ValueError('{}.__spec__ is None'.format(name))
            return spec