def patch_module(module, name, replacement, original=UNSPECIFIED, aliases=True, location=None, **_bogus_options):
    """
    Low-level attribute patcher.

    :param module module: Object to patch.
    :param str name: Attribute to patch
    :param replacement: The replacement value.
    :param original: The original value (in case the object beeing patched uses descriptors or is plain weird).
    :param bool aliases: If ``True`` patch all the attributes that have the same original value.

    :returns: An :obj:`aspectlib.Rollback` object.
    """
    rollback = Rollback()
    seen = False
    original = getattr(module, name) if original is UNSPECIFIED else original
    location = module.__name__ if hasattr(module, '__name__') else type(module).__module__
    target = module.__name__ if hasattr(module, '__name__') else type(module).__name__
    try:
        replacement.__module__ = location
    except (TypeError, AttributeError):
        pass
    for alias in dir(module):
        logdebug("alias:%s (%s)", alias, name)
        if hasattr(module, alias):
            obj = getattr(module, alias)
            logdebug("- %s:%s (%s)", obj, original, obj is original)
            if obj is original:
                if aliases or alias == name:
                    logdebug("= saving %s on %s.%s ...", replacement, target, alias)
                    setattr(module, alias, replacement)
                    rollback.merge(lambda alias=alias: setattr(module, alias, original))
                if alias == name:
                    seen = True
            elif alias == name:
                if ismethod(obj):
                    logdebug("= saving %s on %s.%s ...", replacement, target, alias)
                    setattr(module, alias, replacement)
                    rollback.merge(lambda alias=alias: setattr(module, alias, original))
                    seen = True
                else:
                    raise AssertionError("%s.%s = %s is not %s." % (module, alias, obj, original))

    if not seen:
        warnings.warn('Setting %s.%s to %s. There was no previous definition, probably patching the wrong module.' % (
            target, name, replacement
        ))
        logdebug("= saving %s on %s.%s ...", replacement, target, name)
        setattr(module, name, replacement)
        rollback.merge(lambda: setattr(module, name, original))
    return rollback