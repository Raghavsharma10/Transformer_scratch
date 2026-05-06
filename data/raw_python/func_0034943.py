def install_trigger_function(connection: connection, overwrite: bool=False) -> None:
    """Install the psycopg2-pgevents trigger function against the database.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    overwrite: bool
        Whether or not to overwrite existing installation of psycopg2-pgevents
        trigger function, if existing installation is found.

    Returns
    -------
    None

    """
    prior_install = False

    if not overwrite:
        prior_install = trigger_function_installed(connection)

    if not prior_install:
        log('Installing trigger function...', logger_name=_LOGGER_NAME)

        execute(connection, INSTALL_TRIGGER_FUNCTION_STATEMENT)
    else:
        log('Trigger function already installed; skipping...', logger_name=_LOGGER_NAME)