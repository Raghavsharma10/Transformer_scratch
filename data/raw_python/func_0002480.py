def _run_up(self, path, migration_file, batch, pretend=False):
        """
        Run "up" a migration instance.

        :type migration_file: str

        :type batch: int

        :type pretend: bool
        """
        migration = self._resolve(path, migration_file)

        if pretend:
            return self._pretend_to_run(migration, 'up')

        migration.up()

        self._repository.log(migration_file, batch)

        self._note('<info>✓ Migrated</info> %s' % migration_file)