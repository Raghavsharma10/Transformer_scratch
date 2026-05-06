def select_join(self, table1, table2, cols, table1_col, table2_col=None, join_type=None):
        """
        Left join all rows and columns from two tables where a common value is shared.

        :param table1: Name of table #1
        :param table2: Name of table #2
        :param cols: List of columns or column tuples
            String or flat list: Assumes column(s) are from table #1 if not specified
            List of tuples: Each tuple in list of columns represents (table_name, column_name)
        :param table1_col: Column from table #1 to use as key
        :param table2_col: Column from table #2 to use as key
        :param join_type: Type of join query
        :return: Queried rows
        """
        # Check if cols is a list of tuples
        if isinstance(cols[0], tuple):
            cols = join_cols(['{0}.{1}'.format(tbl, col) for tbl, col in cols])
        else:
            cols = join_cols(['{0}.{1}'.format(table1, col) for col in cols])

        # Validate join_type and table2_col
        join_type = join_type.lower().split(' ', 1)[0].upper() + ' JOIN' if join_type else 'LEFT JOIN'
        assert join_type in JOIN_QUERY_TYPES
        table2_col = table2_col if table2_col else table1_col

        # Concatenate and return statement
        statement = '''
        SELECT {columns}
        FROM {table1}
        {join_type} {table2} ON {table1}.{table1_col} = {table2}.{table2_col}
        '''.format(table1=wrap(table1), table2=wrap(table2), columns=cols, table1_col=table1_col, table2_col=table2_col,
                   join_type=join_type)
        return self.fetch(statement)