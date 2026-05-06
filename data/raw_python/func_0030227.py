def materialize(self, ref, table_name=None, index_columns=None, logger=None):
        """ Creates materialized table for given partition reference.

        Args:
            ref (str): id, vid, name or vname of the partition.

        Returns:
            str: name of the partition table in the database.

        """
        from ambry.library import Library
        assert isinstance(self._library, Library)

        logger.debug('Materializing warehouse partition.\n    partition: {}'.format(ref))
        partition = self._library.partition(ref)

        connection = self._backend._get_connection()

        return self._backend.install(connection, partition, table_name=table_name,
                                     index_columns=index_columns, materialize=True, logger=logger)