def template_inheritance(obj):
    '''
    Generator that iterates the template and its ancestors.
    The order is from most specialized (furthest descendant) to
    most general (furthest ancestor).

    obj can be either:
        1. Mako Template object
        2. Mako `self` object (available within a rendering template)
    '''
    if isinstance(obj, MakoTemplate):
        obj = create_mako_context(obj)['self']
    elif isinstance(obj, MakoContext):
        obj = obj['self']
    while obj is not None:
        yield obj.template
        obj = obj.inherits