def trigger_function_installed(connection: connection):
    """Test whether or not the psycopg2-pgevents trigger function is installed.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.

    Returns
    -------
    bool
        True if the trigger function is installed, otherwise False.

    """
    installed = False

    log('Checking if trigger function installed...', logger_name=_LOGGER_NAME)

    try:
        execute(connection, "SELECT pg_get_functiondef('public.psycopg2_pgevents_create_event'::regproc);")
        installed = True
    except ProgrammingError as e:
        if e.args:
            error_stdout = e.args[0].splitlines()
            error = error_stdout.pop(0)
            if error.endswith('does not exist'):
                # Trigger function not installed
                pass
            else:
                # Some other exception; re-raise
                raise e
        else:
            # Some other exception; re-raise
            raise e

    log('...{}installed'.format('' if installed else 'NOT '), logger_name=_LOGGER_NAME)

    return installed