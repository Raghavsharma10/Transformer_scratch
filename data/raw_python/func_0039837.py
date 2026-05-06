def ensure_environment(variables):
    """
    Check os.environ to ensure that a given collection of
    variables has been set.

    :param variables: A collection of environment variable names
    :returns: os.environ
    :raises IncompleteEnvironment: if any variables are not set, with
        the exception's ``variables`` attribute populated with the
        missing variables
    """
    missing = [v for v in variables if v not in os.environ]
    if missing:
        formatted = ', '.join(missing)
        message = 'Environment variables not set: {}'.format(formatted)
        raise IncompleteEnvironment(message, missing)
    return os.environ