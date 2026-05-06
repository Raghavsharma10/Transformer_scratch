def _get_mpr_table(self, connection, partition):
        """ Returns name of the sqlite table who stores mpr data.

        Args:
            connection (apsw.Connection): connection to sqlite database who stores mpr data.
            partition (orm.Partition):

        Returns:
            str:

        Raises:
            MissingTableError: if partition table not found in the db.

        """
        # TODO: This is the first candidate for optimization. Add field to partition
        # with table name and update it while table creation.
        # Optimized version.
        #
        # return partition.mpr_table or raise exception

        # Not optimized version.
        #
        # first check either partition has readonly table.
        virtual_table = partition.vid
        table = '{}_v'.format(virtual_table)
        logger.debug(
            'Looking for materialized table of the partition.\n    partition: {}'.format(partition.name))
        table_exists = self._relation_exists(connection, table)
        if table_exists:
            logger.debug(
                'Materialized table of the partition found.\n    partition: {}, table: {}'
                .format(partition.name, table))
            return table

        # now check for virtual table
        logger.debug(
            'Looking for a virtual table of the partition.\n    partition: {}'.format(partition.name))
        virtual_exists = self._relation_exists(connection, virtual_table)
        if virtual_exists:
            logger.debug(
                'Virtual table of the partition found.\n    partition: {}, table: {}'
                .format(partition.name, table))
            return virtual_table
        raise MissingTableError('sqlite database does not have table for mpr of {} partition.'
                                .format(partition.vid))