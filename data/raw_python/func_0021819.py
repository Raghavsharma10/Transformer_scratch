def call_plugins(plugins, method, *arg, **kw):
    """Call all method on plugins in list, that define it, with provided
    arguments. The first response that is not None is returned.
    """
    for plug in plugins:
        func = getattr(plug, method, None)
        if func is None:
            continue
        #LOG.debug("call plugin %s: %s", plug.name, method)
        result = func(*arg, **kw)
        if result is not None:
            return result
    return None