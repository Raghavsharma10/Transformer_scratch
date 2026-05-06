def _set_table(table_name):
    """
    Specify the table to work on.
    """
    _State.connection()
    _State.reflect_metadata()
    _State.table = sqlalchemy.Table(table_name, _State.metadata,
                                    extend_existing=True)

    if list(_State.table.columns.keys()) == []:
        _State.table_pending = True
    else:
        _State.table_pending = False