def update(sql, *args, **kwargs):
        """Updates and commits with an insert sql statement, returns the record, but with a small chance of a race
        condition

        @param sql: sql to execute
        @return: The last row inserted
        """
        assert "update" in sql.lower(), 'This function requires an update statement, provided: {}'.format(sql)
        cursor = CoyoteDb.execute_and_commit(sql, *args, **kwargs)

        # now get that id
        last_row_id = cursor.lastrowid

        return last_row_id