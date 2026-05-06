def _add_partition(self, connection, partition):
        """ Creates FDW for the partition.

        Args:
            connection:
            partition (orm.Partition):

        """
        logger.debug('Creating foreign table for partition.\n    partition: {}'.format(partition.name))
        with connection.cursor() as cursor:
            postgres_med.add_partition(cursor, partition.datafile, partition.vid)