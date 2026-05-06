def copy_database_data(self, source, destination, optimized=False):
        """
        Copy the data from one database to another.

        Retrieve existing data from the source database and insert that data into the destination database.
        """
        # Change database to source
        self.enable_printing = False
        self.change_db(source)
        tables = self.tables

        # Copy database data by executing INSERT and SELECT commands in a single query
        if optimized:
            self._copy_database_data_serverside(source, destination, tables)

        # Generate and execute SELECT and INSERT commands
        else:
            self._copy_database_data_clientside(tables, source, destination)

        self.enable_printing = True