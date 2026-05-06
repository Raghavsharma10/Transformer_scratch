def leave_transaction_management(using=None):
    """
    Leaves transaction management for a running thread. A dirty flag is carried
    over to the surrounding block, as a commit will commit all changes, even
    those from outside. (Commits are on connection level.)
    """
    if using is None:
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            connection.leave_transaction_management()
        return
    connection = tldap.backend.connections[using]
    connection.leave_transaction_management()