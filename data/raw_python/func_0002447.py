def compile_create(self, blueprint, command, _):
        """
        Compile a create table command.
        """
        columns = ', '.join(self._get_columns(blueprint))

        sql = 'CREATE TABLE %s (%s' % (self.wrap_table(blueprint), columns)

        sql += self._add_foreign_keys(blueprint)

        sql += self._add_primary_keys(blueprint)

        return sql + ')'