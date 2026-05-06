def load_subclasses(klass, modules=None):
    """Load recursively all all subclasses from a module.

    Args:
        klass (str or list of str): Class whose subclasses we want to load.
        modules: List of additional modules or module names that should be
            recursively imported in order to find all the subclasses of the
            desired class. Default: None

    FIXME: This function is kept only for backward compatibility reasons, it
        should not be used. Deprecation warning should be raised and it should
        be replaces by the ``Loader`` class.
    """
    if modules:
        if isinstance(modules, six.string_types):
            modules = [modules]
        loader = Loader()
        loader.load(*modules)
    return klass.__subclasses__()