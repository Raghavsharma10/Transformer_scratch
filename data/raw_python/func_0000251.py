def is_managed(using=None):
    """
    Checks whether the transaction manager is in manual or in auto state.
    """
    if using is None:
        managed = False
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            if connection.is_managed():
                managed = True
        return managed
    connection = tldap.backend.connections[using]
    return connection.is_managed()