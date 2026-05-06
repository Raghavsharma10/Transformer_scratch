def execute(self, i, o):
        """
        Executes the command.

        :type i: cleo.inputs.input.Input
        :type o: cleo.outputs.output.Output
        """
        super(ResetCommand, self).execute(i, o)

        dialog = self.get_helper('dialog')
        confirm = dialog.ask_confirmation(
            o,
            '<question>Are you sure you want to reset all of the migrations?</question> ',
            False
        )
        if not confirm:
            return

        database = i.get_option('database')
        repository = DatabaseMigrationRepository(self._resolver, 'migrations')

        migrator = Migrator(repository, self._resolver)

        self._prepare_database(migrator, database, i, o)

        pretend = bool(i.get_option('pretend'))

        path = i.get_option('path')

        if path is None:
            path = self._get_migration_path()

        while True:
            count = migrator.rollback(path, pretend)

            for note in migrator.get_notes():
                o.writeln(note)

            if count == 0:
                break