def defaultconfig(cls):
    """
    Generate a default configuration mapping bases on the class name. If this class does not have a
    parent with `configbase` defined, it is set to a configuration base with
    `configbase=<lowercase-name>` and `configkey=<lowercase-name>.default`; otherwise it inherits
    `configbase` of its parent and set `configkey=<parentbase>.<lowercase-name>`
    
    Refer to :ref::`configurations` for normal rules.
    """
    parentbase = None
    for p in cls.__bases__:
        if issubclass(p, Configurable):
            parentbase = getattr(p, 'configbase', None)
            break
    if parentbase is None:
        base = cls.__name__.lower()
        cls.configbase = base
        cls.configkey = base + '.default'
    else:
        key = cls.__name__.lower()
        #=======================================================================
        # parentkeys = parentbase.split('.')
        # for pk in parentkeys:
        #     if key.endswith(pk):
        #         key = key[0:-len(pk)]
        #     elif key.startswith(pk):
        #         key = key[len(pk):]
        #=======================================================================
        cls.configkey = parentbase + "." + key
    return cls