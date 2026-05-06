def execute(self, i, o):
        """
        Executes the command.

        :type i: cleo.inputs.input.Input
        :type o: cleo.outputs.output.Output
        """
        super(InstallCommand, self).execute(i, o)

        database = i.get_option('database')
        repository = DatabaseMigrationRepository(self._resolver, 'migrations')

        repository.set_source(database)
        repository.create_repository()

        o.writeln('<info>Migration table created successfully</info>')