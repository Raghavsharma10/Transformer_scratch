def weave_module(module, aspect, methods=NORMAL_METHODS, lazy=False, bag=BrokenBag, **options):
    """
    Low-level weaver for "whole module weaving".

    .. warning:: You should not use this directly.

    :returns: An :obj:`aspectlib.Rollback` object.
    """
    if bag.has(module):
        return Nothing

    entanglement = Rollback()
    method_matches = make_method_matcher(methods)
    logdebug("weave_module (module=%r, aspect=%s, methods=%s, lazy=%s, **options=%s)",
             module, aspect, methods, lazy, options)

    for attr in dir(module):
        func = getattr(module, attr)
        if method_matches(attr):
            if isroutine(func):
                entanglement.merge(patch_module_function(module, func, aspect, force_name=attr, **options))
            elif isclass(func):
                entanglement.merge(
                    weave_class(func, aspect, owner=module, name=attr, methods=methods, lazy=lazy, bag=bag, **options),
                    #  it's not consistent with the other ways of weaving a class (it's never weaved as a routine).
                    #  therefore it's disabled until it's considered useful.
                    #  #patch_module_function(module, getattr(module, attr), aspect, force_name=attr, **options),
                )
    return entanglement