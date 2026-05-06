def delete(sql, *args, **kwargs):
        """Deletes and commits with an insert sql statement"""
        assert "delete" in sql.lower(), 'This function requires a delete statement, provided: {}'.format(sql)
        CoyoteDb.execute_and_commit(sql, *args, **kwargs)