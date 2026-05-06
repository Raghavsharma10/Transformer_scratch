def get_connection_string(connection=None):
    """return SQLAlchemy connection string if it is set

    :param connection: get the SQLAlchemy connection string #TODO
    :rtype: str
    """
    if not connection:
        config = configparser.ConfigParser()
        cfp = defaults.config_file_path
        if os.path.exists(cfp):
            log.info('fetch database configuration from %s', cfp)
            config.read(cfp)
            connection = config['database']['sqlalchemy_connection_string']
            log.info('load connection string from %s: %s', cfp, connection)
        else:
            with open(cfp, 'w') as config_file:
                connection = defaults.sqlalchemy_connection_string_default
                config['database'] = {'sqlalchemy_connection_string': connection}
                config.write(config_file)
                log.info('create configuration file %s', cfp)

    return connection