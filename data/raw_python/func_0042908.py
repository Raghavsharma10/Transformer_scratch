def deactivate():
    """
    Deactivate a state in this thread.
    """

    if hasattr(_mode, "current_state"):
        del _mode.current_state
    if hasattr(_mode, "schema"):
        del _mode.schema

    for k in connections:
        con = connections[k]
        if hasattr(con, 'reset_schema'):
            con.reset_schema()