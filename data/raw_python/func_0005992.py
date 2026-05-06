def execute_and_commit(*args, **kwargs):
        """Executes and commits the sql statement

        @return: None
        """
        db, cursor = CoyoteDb.execute(*args, **kwargs)
        db.commit()
        return cursor