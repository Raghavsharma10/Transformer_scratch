def setPluginSetting(name, value, namespace = None):
    '''
    Sets the value of a plugin setting.

    :param name: the name of the setting. It is not the full path, but just the last name of it
    :param value: the value to set for the plugin setting
    :param namespace: The namespace. If not passed or None, the namespace will be inferred from
    the caller method. Normally, this should not be passed, since it suffices to let this function
    find out the plugin from where it is being called, and it will automatically use the
    corresponding plugin namespace
    '''
    namespace = namespace or _callerName().split(".")[0]
    settings.setValue(namespace + "/" + name, value)