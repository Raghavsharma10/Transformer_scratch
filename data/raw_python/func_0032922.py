def getDefaultShareID(store):
    """
    Get the highest-priority default share ID for C{store}.

    @return: the default share ID, or u'' if one has not been set.
    @rtype: C{unicode}
    """
    defaultShareID = store.findFirst(
        _DefaultShareID, sort=_DefaultShareID.priority.desc)
    if defaultShareID is None:
        return u''
    return defaultShareID.shareID