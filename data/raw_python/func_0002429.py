def execute(self, i, o):
        """
        Executes the command.

        :type i: cleo.inputs.input.Input
        :type o: cleo.outputs.output.Output
        """
        super(MigrateMakeCommand, self).execute(i, o)

        creator = MigrationCreator()

        name = i.get_argument('name')
        table = i.get_option('table')
        create = bool(i.get_option('create'))

        if not table and create is not False:
            table = create

        path = i.get_option('path')
        if path is None:
            path = self._get_migration_path()

        file_ = self._write_migration(creator, name, table, create, path)

        o.writeln('<info>Create migration: <comment>%s</comment></info>' % file_)