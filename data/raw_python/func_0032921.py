def addDefaultShareID(store, shareID, priority):
    """
    Add a default share ID to C{store}, pointing to C{shareID} with a
    priority C{priority}.  The highest-priority share ID identifies the share
    that will be retrieved when a user does not explicitly provide a share ID
    in their URL (e.g. /host/users/username/).

    @param shareID: A share ID.
    @type shareID: C{unicode}

    @param priority: The priority of this default.  Higher means more
    important.
    @type priority: C{int}
    """
    _DefaultShareID(store=store, shareID=shareID, priority=priority)