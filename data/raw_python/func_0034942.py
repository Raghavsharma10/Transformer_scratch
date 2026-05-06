def trigger_installed(connection: connection, table: str, schema: str='public'):
    """Test whether or not a psycopg2-pgevents trigger is installed for a table.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    table: str
        Table whose trigger-existence will be checked.
    schema: str
        Schema to which the table belongs.

    Returns
    -------
    bool
        True if the trigger is installed, otherwise False.

    """
    installed = False

    log('Checking if {}.{} trigger installed...'.format(schema, table), logger_name=_LOGGER_NAME)

    statement = SELECT_TRIGGER_STATEMENT.format(
        table=table,
        schema=schema
    )

    result = execute(connection, statement)
    if result:
        installed = True

    log('...{}installed'.format('' if installed else 'NOT '), logger_name=_LOGGER_NAME)

    return installed