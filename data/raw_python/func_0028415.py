def check_database_connected(app_configs, **kwargs):
    """
    A Django check to see if connecting to the configured default
    database backend succeeds.
    """
    errors = []

    try:
        connection.ensure_connection()
    except OperationalError as e:
        msg = 'Could not connect to database: {!s}'.format(e)
        errors.append(checks.Error(msg,
                                   id=health.ERROR_CANNOT_CONNECT_DATABASE))
    except ImproperlyConfigured as e:
        msg = 'Datbase misconfigured: "{!s}"'.format(e)
        errors.append(checks.Error(msg,
                                   id=health.ERROR_MISCONFIGURED_DATABASE))
    else:
        if not connection.is_usable():
            errors.append(checks.Error('Database connection is not usable',
                                       id=health.ERROR_UNUSABLE_DATABASE))

    return errors