def copy_database_structure(self, source, destination, tables=None):
        """Copy multiple tables from one database to another."""
        # Change database to source
        self.change_db(source)

        if tables is None:
            tables = self.tables

        # Change database to destination
        self.change_db(destination)
        for t in tqdm(tables, total=len(tables), desc='Copying {0} table structure'.format(source)):
            self.copy_table_structure(source, destination, t)