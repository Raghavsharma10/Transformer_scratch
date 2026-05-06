def uninstall_trigger(connection: connection, table: str, schema: str='public') -> None:
    """Uninstall a psycopg2-pgevents trigger from a table.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    table: str
        Table for which the trigger should be uninstalled.
    schema: str
        Schema to which the table belongs.

    Returns
    -------
    None

    """
    log('Uninstalling {}.{} trigger...'.format(schema, table), logger_name=_LOGGER_NAME)

    statement = UNINSTALL_TRIGGER_STATEMENT.format(
        schema=schema,
        table=table
    )
    execute(connection, statement)