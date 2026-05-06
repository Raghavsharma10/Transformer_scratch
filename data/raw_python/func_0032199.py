def _organizerPluginName(plugin):
    """
    Get a name for C{plugin}, taking into account the fact that it might not
    have defined L{IOrganizerPlugin.name}.

    @type plugin: L{IOrganizerPlugin} provider.

    @rtype: C{unicode}
    """
    name = getattr(plugin, 'name', None)
    if name is not None:
        return name
    warn(
        "IOrganizerPlugin now has the 'name' attribute"
        " and %s does not define it" % (plugin.__class__,),
        category=PendingDeprecationWarning)
    return _objectToName(plugin).decode('ascii')