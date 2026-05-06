def configbase(key):
    """
    Decorator to set this class to configuration base class. A configuration base class
    uses `<parentbase>.key.` for its configuration base, and uses `<parentbase>.key.default` for configuration mapping.
    """
    def decorator(cls):
        parent = cls.getConfigurableParent()
        if parent is None:
            parentbase = None
        else:
            parentbase = getattr(parent, 'configbase', None)
        if parentbase is None:
            base = key
        else:
            base = parentbase + '.' + key
        cls.configbase = base
        cls.configkey = base + '.default'
        return cls
    return decorator