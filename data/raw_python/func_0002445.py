def compile_rename_column(self, blueprint, command, connection):
        """
        Compile a rename column command.

        :param blueprint: The blueprint
        :type blueprint: Blueprint

        :param command: The command
        :type command: Fluent

        :param connection: The connection
        :type connection: eloquent.connections.Connection

        :rtype: list
        """
        # The code is a little complex. It will propably change
        # if we support complete diffs in dbal
        sql = []

        schema = connection.get_schema_manager()
        table = self.get_table_prefix() + blueprint.get_table()

        column = connection.get_column(table, command.from_)

        columns = schema.list_table_columns(table)
        indexes = schema.list_table_indexes(table)
        foreign_keys = schema.list_table_foreign_keys(table)

        diff = self._get_renamed_diff(blueprint, command, column, schema)
        renamed_columns = diff.renamed_columns

        old_column_names = list(map(lambda x: x.get_name(), columns))

        # We build the new column names
        new_column_names = []
        for column_name in old_column_names:
            if column_name in renamed_columns:
                new_column_names.append(renamed_columns[column_name].get_name())
            else:
                new_column_names.append(column_name)

        # We create a temporary table and insert the data into it
        temp_table = '__temp__' + self.get_table_prefix() + blueprint.get_table()
        sql.append('CREATE TEMPORARY TABLE %s AS SELECT %s FROM %s'
                   % (temp_table, self.columnize(old_column_names), table))

        # We drop the current table
        sql += Blueprint(table).drop().to_sql(None, self)

        # Building the list a new columns
        new_columns = []
        for column in columns:
            for column_name, changed_column in renamed_columns.items():
                if column_name == column.get_name():
                    new_columns.append(changed_column)

        # Here we will try to rebuild a new blueprint to create a new table
        # with the original name
        new_blueprint = Blueprint(table)
        new_blueprint.create()
        primary = []
        for column in columns:
            # Mapping the database type to the blueprint type
            type = schema.get_database_platform().TYPE_MAPPING[column.get_type().lower()]

            # If the column is a primary, we will add it to the blueprint later
            if column.get_platform_option('pk'):
                primary.append(column.get_name())

            # If the column is not one that's been renamed we reinsert it into the blueprint
            if column.get_name() not in renamed_columns.keys():
                col = getattr(new_blueprint, type)(column.get_name())

                # If the column is nullable, we flag it
                if not column.get_notnull():
                    col.nullable()

                # If the column has a default value, we add it
                if column.get_default() is not None:
                    col.default(QueryExpression(column.get_default()))

        # Inserting the renamed columns into the blueprint
        for column in new_columns:
            type = schema.get_database_platform().TYPE_MAPPING[column.get_type().lower()]

            col = getattr(new_blueprint, type)(column.get_name())
            if not column.get_notnull():
                col.nullable()

            if column.get_default() is not None:
                col.default(QueryExpression(column.get_default()))

        # We add the primary keys
        if primary:
            new_blueprint.primary(primary)

        # We rebuild the indexes
        for index in indexes:
            index_columns = index['columns']
            new_index_columns = []
            index_name = index['name']

            for column_name in index_columns:
                if column_name in renamed_columns:
                    new_index_columns.append(renamed_columns[column_name].get_name())
                else:
                    new_index_columns.append(column_name)

            if index_columns != new_index_columns:
                index_name = None

            if index['unique']:
                new_blueprint.unique(new_index_columns, index_name)
            else:
                new_blueprint.index(index['columns'], index_name)

        for foreign_key in foreign_keys:
            fkey_from = foreign_key['from']
            if fkey_from in renamed_columns:
                fkey_from = renamed_columns[fkey_from].get_name()

            new_blueprint.foreign(fkey_from)\
                .references(foreign_key['to'])\
                .on(foreign_key['table'])\
                .on_delete(foreign_key['on_delete'])\
                .on_update(foreign_key['on_update'])

        # We create the table
        sql += new_blueprint.to_sql(None, self)

        # We reinsert the data into the new table
        sql.append('INSERT INTO %s (%s) SELECT %s FROM %s'
                   % (self.wrap_table(table),
                      ', '.join(new_column_names),
                      self.columnize(old_column_names),
                      self.wrap_table(temp_table)
                      ))

        # Finally we drop the temporary table
        sql += Blueprint(temp_table).drop().to_sql(None, self)

        return sql