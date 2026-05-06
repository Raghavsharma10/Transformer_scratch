def enter_transaction_management(using=None):
    """
    Enters transaction management for a running thread. It must be balanced
    with the appropriate leave_transaction_management call, since the actual
    state is managed as a stack.

    The state and dirty flag are carried over from the surrounding block or
    from the settings, if there is no surrounding block (dirty is always false
    when no current block is running).
    """
    if using is None:
        for using in tldap.backend.connections:
            connection = tldap.backend.connections[using]
            connection.enter_transaction_management()
        return
    connection = tldap.backend.connections[using]
    connection.enter_transaction_management()