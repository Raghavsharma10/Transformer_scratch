def update_many(self, table, columns, values, where_col, where_index):
        """
        Update the values of several rows.

        :param table: Name of the MySQL table
        :param columns: List of columns
        :param values: 2D list of rows
        :param where_col: Column name for where clause
        :param where_index: Row index of value to be used for where comparison
        :return:
        """
        for row in values:
            wi = row.pop(where_index)
            self.update(table, columns, row, (where_col, wi))