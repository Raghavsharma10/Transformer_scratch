def rollback(self, path, pretend=False):
        """
        Rollback the last migration operation.

        :param path: The path
        :type path: str

        :param pretend: Whether we execute the migrations as dry-run
        :type pretend: bool

        :rtype: int
        """
        self._notes = []

        migrations = self._repository.get_last()

        if not migrations:
            self._note('<info>Nothing to rollback.</info>')

            return len(migrations)

        for migration in migrations:
            self._run_down(path, migration, pretend)

        return len(migrations)