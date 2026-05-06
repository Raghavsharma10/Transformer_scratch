def pluginSetting(name, namespace=None, typ=None):
    '''
    Returns the value of a plugin setting.

    :param name: the name of the setting. It is not the full path, but just the last name of it
    :param namespace: The namespace. If not passed or None, the namespace will be inferred from
    the caller method. Normally, this should not be passed, since it suffices to let this function
    find out the plugin from where it is being called, and it will automatically use the
    corresponding plugin namespace
    '''
    def _find_in_cache(name, key):
        for setting in _settings[namespace]:
            if setting["name"] == name:
                return setting[key]
        return None

    def _type_map(t):
        """Return setting python type"""
        if t == BOOL:
            return bool
        elif t == NUMBER:
            return float
        else:
            return unicode

    namespace = namespace or _callerName().split(".")[0]
    full_name = namespace + "/" + name
    if settings.contains(full_name):
        if typ is None:
            typ = _type_map(_find_in_cache(name, 'type'))
        v = settings.value(full_name, None, type=typ)
        try:
            if isinstance(v, QPyNullVariant):
                v = None
        except:
            pass
        return v
    else:
        return _find_in_cache(name, 'default')