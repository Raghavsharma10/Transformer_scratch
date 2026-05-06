def _compare_dbs_getter(self, db):
        """Retrieve a dictionary of table_name, row count key value pairs for a DB."""
        # Change DB connection if needed
        if self.database != db:
            self.change_db(db)
        return self.count_rows_all()