def get_alter_table_sql(self, diff):
        """
        Get the ALTER TABLE SQL statement

        :param diff: The table diff
        :type diff: eloquent.dbal.table_diff.TableDiff

        :rtype: list
        """
        column_sql = []
        query_parts = []

        if diff.new_name is not False:
            query_parts.append('RENAME TO %s' % diff.new_name)

        # Added columns?

        # Removed columns?

        for column_diff in diff.changed_columns.values():
            column = column_diff.column
            column_dict = column.to_dict()

            # Don't propagate default value changes for unsupported column types.
            if column_diff.has_changed('default') \
                    and len(column_diff.changed_properties) == 1 \
                    and (column_dict['type'] == 'text' or column_dict['type'] == 'blob'):
                continue

            query_parts.append('CHANGE %s %s'
                               % (column_diff.get_old_column_name(),
                                  self.get_column_declaration_sql(column.get_name(), column_dict)))

        for old_column_name, column in diff.renamed_columns.items():
            column_dict = column.to_dict()
            query_parts.append('CHANGE %s %s'
                               % (self.quote(old_column_name),
                                  self.get_column_declaration_sql(self.quote(column.get_name()), column_dict)))

        sql = []

        if len(query_parts) > 0:
            sql.append('ALTER TABLE %s %s' % (diff.name, ', '.join(query_parts)))

        return sql