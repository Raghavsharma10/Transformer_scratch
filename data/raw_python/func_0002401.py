def compile_foreign(self, blueprint, command, _):
        """
        Compile a foreign key command.

        :param blueprint: The blueprint
        :type blueprint: Blueprint

        :param command: The command
        :type command: Fluent

        :rtype: str
        """
        table = self.wrap_table(blueprint)

        on = self.wrap_table(command.on)

        columns = self.columnize(command.columns)

        on_columns = self.columnize(command.references
                                    if isinstance(command.references, list)
                                    else [command.references])

        sql = 'ALTER TABLE %s ADD CONSTRAINT %s ' % (table, command.index)

        sql += 'FOREIGN KEY (%s) REFERENCES %s (%s)' % (columns, on, on_columns)

        if command.get('on_delete'):
            sql += ' ON DELETE %s' % command.on_delete

        if command.get('on_update'):
            sql += ' ON UPDATE %s' % command.on_update

        return sql