def install(self, connection, partition, table_name=None, columns=None, materialize=False,
                logger=None):
        """ Creates FDW or materialize view for given partition.

        Args:
            connection: connection to postgresql
            partition (orm.Partition):
            materialize (boolean): if True, create read-only table. If False create virtual table.

        Returns:
            str: name of the created table.

        """

        partition.localize()

        self._add_partition(connection, partition)
        fdw_table = partition.vid
        view_table = '{}_v'.format(fdw_table)

        if materialize:
            with connection.cursor() as cursor:
                view_exists = self._relation_exists(connection, view_table)
                if view_exists:
                    logger.debug(
                        'Materialized view of the partition already exists.\n    partition: {}, view: {}'
                        .format(partition.name, view_table))
                else:
                    query = 'CREATE MATERIALIZED VIEW {} AS SELECT * FROM {};'\
                        .format(view_table, fdw_table)
                    logger.debug(
                        'Creating new materialized view of the partition.'
                        '\n    partition: {}, view: {}, query: {}'
                        .format(partition.name, view_table, query))
                    cursor.execute(query)
                    cursor.execute('COMMIT;')

        final_table = view_table if materialize else fdw_table

        with connection.cursor() as cursor:
            view_q = "CREATE VIEW IF NOT EXISTS {} AS SELECT * FROM {} ".format(partition.vid, final_table)
            cursor.execute(view_q)
            cursor.execute('COMMIT;')

        return partition.vid