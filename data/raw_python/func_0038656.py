def dump_table(self, table, drop_statement=True):
        """Export a table structure and data to SQL file for backup or later import."""
        create_statement = self.get_table_definition(table)
        data = self.select_all(table)
        statements = ['\n', sql_file_comment(''),
                      sql_file_comment('Table structure and data dump for {0}'.format(table)), sql_file_comment('')]
        if drop_statement:
            statements.append('\nDROP TABLE IF EXISTS {0};'.format(wrap(table)))
        statements.append('{0};\n'.format(create_statement))
        if len(data) > 0:
            statements.append('{0};'.format(insert_statement(table, self.get_columns(table), data)))
        return '\n'.join(statements)