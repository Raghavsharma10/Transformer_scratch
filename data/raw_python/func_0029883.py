def _add_partition(self, connection, partition):
        """ Creates sqlite virtual table for mpr file of the given partition.

        Args:
            connection: connection to the sqlite db who stores mpr data.
            partition (orm.Partition):

        """
        logger.debug('Creating virtual table for partition.\n    partition: {}'.format(partition.name))
        sqlite_med.add_partition(connection, partition.datafile, partition.vid+'_vt')