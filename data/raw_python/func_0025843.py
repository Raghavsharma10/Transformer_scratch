def weave_instance(instance, aspect, methods=NORMAL_METHODS, lazy=False, bag=BrokenBag, **options):
    """
    Low-level weaver for instances.

    .. warning:: You should not use this directly.

    :returns: An :obj:`aspectlib.Rollback` object.
    """
    if bag.has(instance):
        return Nothing

    entanglement = Rollback()
    method_matches = make_method_matcher(methods)
    logdebug("weave_instance (module=%r, aspect=%s, methods=%s, lazy=%s, **options=%s)",
             instance, aspect, methods, lazy, options)

    def fixup(func):
        return func.__get__(instance, type(instance))
    fixed_aspect = aspect + [fixup] if isinstance(aspect, (list, tuple)) else [aspect, fixup]

    for attr in dir(instance):
        func = getattr(instance, attr)
        if method_matches(attr):
            if ismethod(func):
                if hasattr(func, '__func__'):
                    realfunc = func.__func__
                else:
                    realfunc = func.im_func
                entanglement.merge(
                    patch_module(instance, attr, _checked_apply(fixed_aspect, realfunc, module=None), **options)
                )
    return entanglement