def rollback(using=None):
    """
    This function does the rollback itself and resets the dirty flag.
    """
    if using is None:
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            connection.rollback()
        return
    connection = tldap.backend.connections[using]
    connection.rollback()