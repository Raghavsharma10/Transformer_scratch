def add_column(connection, column):
    """
    Add a column to the current table.
    """
    stmt = alembic.ddl.base.AddColumn(_State.table.name, column)
    connection.execute(stmt)
    _State.reflect_metadata()