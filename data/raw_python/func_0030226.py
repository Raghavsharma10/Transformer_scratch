def install(self, ref, table_name=None, index_columns=None,logger=None):
        """ Finds partition by reference and installs it to warehouse db.

        Args:
            ref (str): id, vid (versioned id), name or vname (versioned name) of the partition.

        """


        try:
            obj_number = ObjectNumber.parse(ref)
            if isinstance(obj_number, TableNumber):
                table = self._library.table(ref)
                connection = self._backend._get_connection()
                return self._backend.install_table(connection, table, logger=logger)
            else:
                # assume partition
                raise NotObjectNumberError

        except NotObjectNumberError:
            # assume partition.
            partition = self._library.partition(ref)
            connection = self._backend._get_connection()

            return self._backend.install(
                connection, partition, table_name=table_name, index_columns=index_columns,
                logger=logger)