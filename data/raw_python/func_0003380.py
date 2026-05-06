def config(key):
    """
    Decorator to map this class directly to a configuration node. It uses `<parentbase>.key` for configuration
    base and configuration mapping.
    """
    def decorator(cls):
        parent = cls.getConfigurableParent()
        if parent is None:
            parentbase = None
        else:
            parentbase = getattr(parent, 'configbase', None)
        if parentbase is None:
            cls.configkey = key
        else:
            cls.configkey = parentbase + '.' + key
        return cls
    return decorator