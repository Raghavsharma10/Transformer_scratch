def _run_down(self, path, migration, pretend=False):
        """
        Run "down" a migration instance.
        """
        migration_file = migration['migration']

        instance = self._resolve(path, migration_file)

        if pretend:
            return self._pretend_to_run(instance, 'down')

        instance.down()

        self._repository.delete(migration)

        self._note('<info>✓ Rolled back</info> %s' % migration_file)