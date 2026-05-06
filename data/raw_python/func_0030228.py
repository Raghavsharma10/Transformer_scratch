def index(self, ref, columns):
        """ Create an index on the columns.

        Args:
            ref (str): id, vid, name or versioned name of the partition.
            columns (list of str): names of the columns needed indexes.

        """
        from ambry.orm.exc import NotFoundError

        logger.debug('Creating index for partition.\n    ref: {}, columns: {}'.format(ref, columns))

        connection = self._backend._get_connection()

        try:
            table_or_partition = self._library.partition(ref)
        except NotFoundError:
            table_or_partition = ref


        self._backend.index(connection, table_or_partition, columns)