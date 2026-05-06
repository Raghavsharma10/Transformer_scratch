def execute(self, i, o):
        """
        Executes the command.

        :type i: cleo.inputs.input.Input
        :type o: cleo.outputs.output.Output
        """
        super(StatusCommand, self).execute(i, o)

        database = i.get_option('database')
        repository = DatabaseMigrationRepository(self._resolver, 'migrations')

        migrator = Migrator(repository, self._resolver)

        if not migrator.repository_exists():
            return o.writeln('<error>No migrations found</error>')

        self._prepare_database(migrator, database, i, o)

        path = i.get_option('path')

        if path is None:
            path = self._get_migration_path()

        ran = migrator.get_repository().get_ran()

        migrations = []
        for migration in migrator._get_migration_files(path):
            if migration in ran:
                migrations.append(['<fg=cyan>%s</>' % migration, '<info>Yes</info>'])
            else:
                migrations.append(['<fg=cyan>%s</>' % migration, '<fg=red>No</>'])

        if migrations:
            table = self.get_helper('table')
            table.set_headers(['Migration', 'Ran?'])
            table.set_rows(migrations)
            table.render(o)
        else:
            return o.writeln('<error>No migrations found</error>')

        for note in migrator.get_notes():
            o.writeln(note)