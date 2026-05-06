def is_dirty(using=None):
    """
    Returns True if the current transaction requires a commit for changes to
    happen.
    """
    if using is None:
        dirty = False
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            if connection.is_dirty():
                dirty = True
        return dirty
    connection = tldap.backend.connections[using]
    return connection.is_dirty()