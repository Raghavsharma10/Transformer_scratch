def install_table(self, connection, table, logger = None):
        """ Installs all partitons of the table and create view with union of all partitons.

        Args:
            connection: connection to database who stores mpr data.
            table (orm.Table):
        """
        # first install all partitions of the table

        queries = []
        query_tmpl = 'SELECT * FROM {}'
        for partition in table.partitions:
            partition.localize()
            installed_name = self.install(connection, partition)
            queries.append(query_tmpl.format(installed_name))

        # now create view with union of all partitions.
        query = 'CREATE VIEW {} AS {} '.format( table.vid, '\nUNION ALL\n'.join(queries))
        logger.debug('Creating view for table.\n    table: {}\n    query: {}'.format(table.vid, query))
        self._execute(connection, query, fetch=False)