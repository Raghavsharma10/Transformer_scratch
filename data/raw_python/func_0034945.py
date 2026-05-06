def install_trigger(connection: connection, table: str, schema: str='public', overwrite: bool=False) -> None:
    """Install a psycopg2-pgevents trigger against a table.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    table: str
        Table for which the trigger should be installed.
    schema: str
        Schema to which the table belongs.
    overwrite: bool
        Whether or not to overwrite existing installation of trigger for the
        given table, if existing installation is found.

    Returns
    -------
    None

    """
    prior_install = False

    if not overwrite:
        prior_install = trigger_installed(connection, table, schema)

    if not prior_install:
        log('Installing {}.{} trigger...'.format(schema, table), logger_name=_LOGGER_NAME)

        statement = INSTALL_TRIGGER_STATEMENT.format(
            schema=schema,
            table=table
        )
        execute(connection, statement)
    else:
        log('{}.{} trigger already installed; skipping...'.format(schema, table), logger_name=_LOGGER_NAME)