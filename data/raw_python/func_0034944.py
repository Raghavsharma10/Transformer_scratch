def uninstall_trigger_function(connection: connection, force: bool=False) -> None:
    """Uninstall the psycopg2-pgevents trigger function from the database.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    force: bool
        If True, force the un-registration even if dependent triggers are still
        installed. If False, if there are any dependent triggers for the trigger
        function, the un-registration will fail.

    Returns
    -------
    None

    """
    modifier = ''
    if force:
        modifier = 'CASCADE'

    log('Uninstalling trigger function (cascade={})...'.format(force), logger_name=_LOGGER_NAME)

    statement = UNINSTALL_TRIGGER_FUNCTION_STATEMENT.format(modifier=modifier)
    execute(connection, statement)