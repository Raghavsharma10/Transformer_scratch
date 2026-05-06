def _get_queries(self, migration, method):
        """
        Get all of the queries that would be run for a migration.

        :param migration: The migration
        :type migration: eloquent.migrations.migration.Migration

        :param method: The method to execute
        :type method: str

        :rtype: list
        """
        connection = migration.get_connection()

        db = self.resolve_connection(connection)

        logger = logging.getLogger('eloquent.connection.queries')
        level = logger.level
        logger.setLevel(logging.DEBUG)
        handler = MigratorHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        db.pretend(lambda _: getattr(migration, method)())

        logger.removeHandler(handler)
        logger.setLevel(level)

        return handler.queries