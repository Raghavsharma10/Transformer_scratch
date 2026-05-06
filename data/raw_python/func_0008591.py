def obj_classes_from_module(module):
    """Return a list of classes in a module that have a 'classID' attribute."""
    for name in dir(module):
        if not name.startswith('_'):
            cls = getattr(module, name)
            if getattr(cls, 'classID', None):
                yield (name, cls)