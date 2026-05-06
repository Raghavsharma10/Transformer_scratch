def get_alter_table_sql(self, diff):
        """
        Get the ALTER TABLE SQL statement

        :param diff: The table diff
        :type diff: eloquent.dbal.table_diff.TableDiff

        :rtype: list
        """
        #sql = self._get_simple_alter_table_sql(diff)

        from_table = diff.from_table
        if not isinstance(from_table, Table):
            raise Exception('SQLite platform requires for the alter table the table diff '
                            'referencing the original table')

        table = from_table.clone()
        columns = {}
        old_column_names = {}
        new_column_names = {}
        column_sql = []
        for column_name, column in table.get_columns().items():
            column_name = column_name.lower()
            columns[column_name] = column
            old_column_names[column_name] = column.get_name()
            new_column_names[column_name] = column.get_name()

        for column_name, column in diff.removed_columns.items():
            column_name = column_name.lower()
            if column_name in columns:
                del columns[column_name]
                del old_column_names[column_name]
                del new_column_names[column_name]

        for old_column_name, column in diff.renamed_columns.items():
            old_column_name = old_column_name.lower()
            if old_column_name in columns:
                del columns[old_column_name]

            columns[column.get_name().lower()] = column

            if old_column_name in new_column_names:
                new_column_names[old_column_name] = column.get_name()

        for old_column_name, column_diff in diff.changed_columns.items():
            if old_column_name in columns:
                del columns[old_column_name]

            columns[column_diff.column.get_name().lower()] = column_diff.column

            if old_column_name in new_column_names:
                new_column_names[old_column_name] = column_diff.column.get_name()

        for column_name, column in diff.added_columns.items():
            columns[column_name.lower()] = column

        sql = []
        table_sql = []

        data_table = Table('__temp__' + table.get_name())

        new_table = Table(table.get_name(), columns,
                          self.get_primary_index_in_altered_table(diff),
                          self.get_foreign_keys_in_altered_table(diff))
        new_table.add_option('alter', True)

        sql = self.get_pre_alter_table_index_foreign_key_sql(diff)
        sql.append('CREATE TEMPORARY TABLE %s AS SELECT %s FROM %s'
                   % (data_table.get_name(), ', '.join(old_column_names.values()), table.get_name()))
        sql.append(self.get_drop_table_sql(from_table))

        sql += self.get_create_table_sql(new_table)
        sql.append('INSERT INTO %s (%s) SELECT %s FROM %s'
                   % (new_table.get_name(),
                      ', '.join(new_column_names.values()),
                      ', '.join(old_column_names.values()),
                      data_table.get_name()))
        sql.append(self.get_drop_table_sql(data_table))

        sql += self.get_post_alter_table_index_foreign_key_sql(diff)

        return sql