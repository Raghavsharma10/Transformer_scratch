def compile_change(self, blueprint, command, connection):
        """
        Compile a change column command into a series of SQL statement.

        :param blueprint: The blueprint
        :type blueprint: eloquent.schema.Blueprint

        :param command: The command
        :type command: Fluent

        :param connection: The connection
        :type connection: eloquent.connections.Connection

        :rtype: list
        """
        sql = []

        schema = connection.get_schema_manager()
        table = self.get_table_prefix() + blueprint.get_table()

        columns = schema.list_table_columns(table)
        indexes = schema.list_table_indexes(table)
        foreign_keys = schema.list_table_foreign_keys(table)

        diff = self._get_changed_diff(blueprint, schema)
        blueprint_changed_columns = blueprint.get_changed_columns()
        changed_columns = diff.changed_columns

        temp_table = '__temp__' + self.get_table_prefix() + blueprint.get_table()
        sql.append('CREATE TEMPORARY TABLE %s AS SELECT %s FROM %s'
                   % (temp_table, self.columnize(list(map(lambda x: x.get_name(), columns))), table))
        sql += Blueprint(table).drop().to_sql(None, self)

        new_columns = []
        for column in columns:
            for column_name, changed_column in changed_columns.items():
                if column_name == column.get_name():
                    for blueprint_column in blueprint_changed_columns:
                        if blueprint_column.name == column_name:
                            new_columns.append(blueprint_column)
                            break

                    break

        new_blueprint = Blueprint(table)
        new_blueprint.create()
        primary = []
        new_column_names = []
        for column in columns:
            type = schema.get_database_platform().TYPE_MAPPING[column.get_type().lower()]

            if column.get_platform_option('pk'):
                primary.append(column.get_name())

            if column.get_name() not in changed_columns:
                col = getattr(new_blueprint, type)(column.get_name())
                if not column.get_notnull():
                    col.nullable()

                new_column_names.append(column.get_name())

        for column in new_columns:
            column.change = False
            new_blueprint._add_column(**column.get_attributes())
            new_column_names.append(column.name)

        if primary:
            new_blueprint.primary(primary)

        for index in indexes:
            if index['unique']:
                new_blueprint.unique(index['columns'], index['name'])
            else:
                new_blueprint.index(index['columns'], index['name'])

        for foreign_key in foreign_keys:
            new_blueprint.foreign(foreign_key['from'])\
                .references(foreign_key['to'])\
                .on(foreign_key['table'])\
                .on_delete(foreign_key['on_delete'])\
                .on_update(foreign_key['on_update'])

        sql += new_blueprint.to_sql(None, self)
        sql.append('INSERT INTO %s (%s) SELECT %s FROM %s'
                   % (self.wrap_table(table),
                      ', '.join(sorted(new_column_names)),
                      self.columnize(sorted(list(map(lambda x: x.get_name(), columns)))),
                      self.wrap_table(temp_table)
                      ))
        sql += Blueprint(temp_table).drop().to_sql(None, self)

        return sql