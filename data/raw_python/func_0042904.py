def update(self, table_name, where_slice, new_values):
        """
        where_slice - A Data WHICH WILL BE USED TO MATCH ALL IN table
                      eg {"id": 42}
        new_values  - A dict WITH COLUMN NAME, COLUMN VALUE PAIRS TO SET
        """
        new_values = quote_param(new_values)

        where_clause = SQL_AND.join([
            quote_column(k) + "=" + quote_value(v) if v != None else quote_column(k) + SQL_IS_NULL
            for k, v in where_slice.items()
        ])

        command = (
            "UPDATE " + quote_column(table_name) + "\n" +
            "SET " +
            sql_list([quote_column(k) + "=" + v for k, v in new_values.items()]) +
            SQL_WHERE +
            where_clause
        )
        self.execute(command, {})