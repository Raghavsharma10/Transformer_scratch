def _clear_namespace():
    """ Clear names that are not part of the strict ES API
    """
    ok_names = set(default_backend.__dict__)
    ok_names.update(['gl2', 'glplus'])  # don't remove the module
    NS = globals()
    for name in list(NS.keys()):
        if name.lower().startswith('gl'):
            if name not in ok_names:
                del NS[name]