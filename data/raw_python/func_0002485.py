def _resolve(self, path, migration_file):
        """
        Resolve a migration instance from a file.

        :param migration_file: The migration file
        :type migration_file: str

        :rtype: eloquent.migrations.migration.Migration
        """
        variables = {}

        name = '_'.join(migration_file.split('_')[4:])
        migration_file = os.path.join(path, '%s.py' % migration_file)

        with open(migration_file) as fh:
            exec(fh.read(), {}, variables)

        klass = variables[inflection.camelize(name)]

        instance = klass()
        instance.set_schema_builder(self.get_repository().get_connection().get_schema_builder())

        return instance