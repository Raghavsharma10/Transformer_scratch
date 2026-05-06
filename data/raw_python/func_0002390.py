def get_alter_table_sql(self, diff):
        """
        Get the ALTER TABLE SQL statement

        :param diff: The table diff
        :type diff: eloquent.dbal.table_diff.TableDiff

        :rtype: list
        """
        sql = []

        for column_diff in diff.changed_columns.values():
            if self.is_unchanged_binary_column(column_diff):
                continue

            old_column_name = column_diff.old_column_name
            column = column_diff.column

            if any([column_diff.has_changed('type'),
                    column_diff.has_changed('precision'),
                    column_diff.has_changed('scale'),
                    column_diff.has_changed('fixed')]):
                query = 'ALTER ' + old_column_name + ' TYPE ' + self.get_sql_type_declaration(column.to_dict())
                sql.append('ALTER TABLE ' + diff.name + ' ' + query)

            if column_diff.has_changed('default') or column_diff.has_changed('type'):
                if column.get_default() is None:
                    default_clause = ' DROP DEFAULT'
                else:
                    default_clause = ' SET' + self.get_default_value_declaration_sql(column.to_dict())

                query = 'ALTER ' + old_column_name + default_clause
                sql.append('ALTER TABLE ' + diff.name + ' ' + query)

            if column_diff.has_changed('notnull'):
                op = 'DROP'
                if column.get_notnull():
                    op = 'SET'

                query = 'ALTER ' + old_column_name + ' ' + op + ' NOT NULL'
                sql.append('ALTER TABLE ' + diff.name + ' ' + query)

            if column_diff.has_changed('autoincrement'):
                if column.get_autoincrement():
                    seq_name = self.get_identity_sequence_name(diff.name, old_column_name)

                    sql.append('CREATE SEQUENCE ' + seq_name)
                    sql.append('SELECT setval(\'' + seq_name + '\', '
                               '(SELECT MAX(' + old_column_name + ') FROM ' + diff.name + '))')
                    query = 'ALTER ' + old_column_name + ' SET DEFAULT nextval(\'' + seq_name + '\')'
                    sql.append('ALTER TABLE ' + diff.name + ' ' + query)
                else:
                    query = 'ALTER ' + old_column_name + ' DROP DEFAULT'
                    sql.append('ALTER TABLE ' + diff.name + ' ' + query)

            if column_diff.has_changed('length'):
                query = 'ALTER ' + old_column_name + ' TYPE ' + self.get_sql_type_declaration(column.to_dict())
                sql.append('ALTER TABLE ' + diff.name + ' ' + query)

        for old_column_name, column in diff.renamed_columns.items():
            sql.append('ALTER TABLE ' + diff.name + ' '
                       'RENAME COLUMN ' + old_column_name + ' TO ' + column.get_name())

        return sql