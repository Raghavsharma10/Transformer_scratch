def dump_database(self, file_path, database=None, tables=None):
        """
        Export the table structure and data for tables in a database.

        If not database is specified, it is assumed the currently connected database
        is the source.  If no tables are provided, all tables will be dumped.
        """
        # Change database if needed
        if database:
            self.change_db(database)

        # Set table
        if not tables:
            tables = self.tables

        # Retrieve and join dump statements
        statements = [self.dump_table(table) for table in tqdm(tables, total=len(tables), desc='Generating dump files')]
        dump = 'SET FOREIGN_KEY_CHECKS=0;' + '\n'.join(statements) + '\nSET FOREIGN_KEY_CHECKS=1;'

        # Write dump statements to sql file
        file_path = file_path if file_path.endswith('.sql') else file_path + '.sql'
        write_text(dump, file_path)
        return file_path