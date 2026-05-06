def getenv(variable_key, default=None, required=False, silent=True):
    '''getenv will attempt to get an environment variable. If the variable
    is not found, None is returned.
    :param variable_key: the variable name
    :param required: exit with error if not found
    :param silent: Do not print debugging information for variable
    '''
    variable = os.environ.get(variable_key, default)
    if variable is None and required:
        bot.error("Cannot find environment variable %s, exiting." %variable_key)
        sys.exit(1)

    if not silent:
        if variable is not None:
            bot.verbose2("%s found as %s" %(variable_key,variable))
        else:
            bot.verbose2("%s not defined (None)" %variable_key)

    return variable