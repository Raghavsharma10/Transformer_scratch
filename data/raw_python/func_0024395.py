def migrate(config):
    """Perform a migration according to config.

    :param config: The configuration to be applied
    :type config: Config
    """
    webapp = WebApp(config.web_host, config.web_port,
                    custom_maintenance_file=config.web_custom_html)

    webserver = WebServer(webapp)
    webserver.daemon = True
    webserver.start()

    migration_parser = YamlParser.parse_from_file(config.migration_file)
    migration = migration_parser.parse()

    database = Database(config)

    with database.connect() as lock_connection:
        application_lock = ApplicationLock(lock_connection)
        application_lock.start()

        while not application_lock.acquired:
            time.sleep(0.5)
        else:
            if application_lock.replica:
                # when a replica could finally acquire a lock, it
                # means that the concurrent process has finished the
                # migration or that it failed to run it.
                # In both cases after the lock is released, this process will
                # verify if it has still to do something (if the other process
                # failed mainly).
                application_lock.stop = True
                application_lock.join()
            # we are not in the replica or the lock is released: go on for the
            # migration

        try:
            table = MigrationTable(database)
            runner = Runner(config, migration, database, table)
            runner.perform()
        finally:
            application_lock.stop = True
            application_lock.join()