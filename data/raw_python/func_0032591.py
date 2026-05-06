def _realGetLoader(n, default=_marker):
    """
    Search all themes for a template named C{n}, returning a loader
    for it if found. If not found and a default is passed, the default
    will be returned. Otherwise C{RuntimeError} will be raised.

    This function is deprecated in favor of using a L{ThemedElement}
    for your view code, or calling
    ITemplateNameResolver(userStore).getDocFactory.
    """
    for t in getAllThemes():
        fact = t.getDocFactory(n, None)
        if fact is not None:
            return fact
    if default is _marker:
        raise RuntimeError("No loader for %r anywhere" % (n,))
    return default