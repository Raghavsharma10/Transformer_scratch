def _get_mpr_table(self, connection, partition):
        """ Returns name of the postgres table who stores mpr data.

        Args:
            connection: connection to postgres db who stores mpr data.
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
        # first check either partition has materialized view.
        logger.debug(
            'Looking for materialized view of the partition.\n    partition: {}'.format(partition.name))
        foreign_table = partition.vid
        view_table = '{}_v'.format(foreign_table)
        view_exists = self._relation_exists(connection, view_table)
        if view_exists:
            logger.debug(
                'Materialized view of the partition found.\n    partition: {}, view: {}'
                .format(partition.name, view_table))
            return view_table

        # now check for fdw/virtual table
        logger.debug(
            'Looking for foreign table of the partition.\n    partition: {}'.format(partition.name))
        foreign_exists = self._relation_exists(connection, foreign_table)
        if foreign_exists:
            logger.debug(
                'Foreign table of the partition found.\n    partition: {}, foreign table: {}'
                .format(partition.name, foreign_table))
            return foreign_table
        raise MissingTableError('postgres database does not have table for {} partition.'
                                .format(partition.vid))