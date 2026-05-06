def commit(using=None):
    """
    Does the commit itself and resets the dirty flag.
    """
    if using is None:
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            connection.commit()
        return
    connection = tldap.backend.connections[using]
    connection.commit()