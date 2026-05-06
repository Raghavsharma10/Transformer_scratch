def get_database_columns(self, tables=None, database=None):
        """Retrieve a dictionary of columns."""
        # Get table data and columns from source database
        source = database if database else self.database
        tables = tables if tables else self.tables
        return {tbl: self.get_columns(tbl) for tbl in tqdm(tables, total=len(tables),
                                                           desc='Getting {0} columns'.format(source))}