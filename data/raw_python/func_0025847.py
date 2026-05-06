def patch_module_function(module, target, aspect, force_name=None, bag=BrokenBag, **options):
    """
    Low-level patcher for one function from a specified module.

    .. warning:: You should not use this directly.

    :returns: An :obj:`aspectlib.Rollback` object.
    """
    logdebug("patch_module_function (module=%s, target=%s, aspect=%s, force_name=%s, **options=%s",
             module, target, aspect, force_name, options)
    name = force_name or target.__name__
    return patch_module(module, name, _checked_apply(aspect, target, module=module), original=target, **options)