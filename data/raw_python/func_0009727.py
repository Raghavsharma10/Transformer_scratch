def drop():
    """
    Drop the current table if it exists
    """
    # Ensure the connection is up
    _State.connection()
    _State.table.drop(checkfirst=True)
    _State.metadata.remove(_State.table)
    _State.table = None
    _State.new_transaction()