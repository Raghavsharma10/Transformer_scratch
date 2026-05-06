def create_table(unique_keys):
    """
    Save the table currently waiting to be created.
    """
    _State.new_transaction()
    _State.table.create(bind=_State.engine, checkfirst=True)
    if unique_keys != []:
        create_index(unique_keys, unique=True)
    _State.table_pending = False
    _State.reflect_metadata()