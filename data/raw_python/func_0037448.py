def set_connection(connection=defaults.sqlalchemy_connection_string_default):
    """Set the connection string for SQLAlchemy

    :param str connection: SQLAlchemy connection string
    """
    cfp = defaults.config_file_path
    config = RawConfigParser()

    if not os.path.exists(cfp):
        with open(cfp, 'w') as config_file:
            config['database'] = {'sqlalchemy_connection_string': connection}
            config.write(config_file)
            log.info('create configuration file %s', cfp)
    else:
        config.read(cfp)
        config.set('database', 'sqlalchemy_connection_string', connection)
        with open(cfp, 'w') as configfile:
            config.write(configfile)