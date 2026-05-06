def copy_table_structure(self, source, destination, table):
        """
        Copy a table from one database to another.

        :param source: Source database
        :param destination: Destination database
        :param table: Table name
        """
        self.execute('CREATE TABLE {0}.{1} LIKE {2}.{1}'.format(destination, wrap(table), source))