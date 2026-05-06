def _pretend_to_run(self, migration, method):
        """
        Pretend to run the migration.

        :param migration: The migration
        :type migration: eloquent.migrations.migration.Migration

        :param method: The method to execute
        :type method: str
        """
        for query in self._get_queries(migration, method):
            name = migration.__class__.__name__

            self._note('<info>%s:</info> <comment>%s</comment>' % (name, query))