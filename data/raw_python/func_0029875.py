def install(self, connection, partition, table_name = None, index_columns=None, materialize=False,
                logger = None):
        """ Creates virtual table or read-only table for gion.

        Args:
            ref (str): id, vid, name or versioned name of the partition.
            materialize (boolean): if True, create read-only table. If False create virtual table.

        Returns:
            str: name of the created table.

        """
        virtual_table = partition.vid

        table = partition.vid if not table_name else table_name

        if self._relation_exists(connection, table):
            if logger:
                logger.debug("Skipping '{}'; already installed".format(table))
            return
        else:
            if logger:
                logger.info("Installing '{}'".format(table))

        partition.localize()


        virtual_table = partition.vid + '_vt'

        self._add_partition(connection, partition)

        if materialize:

            if self._relation_exists(connection, table):
                debug_logger.debug(
                    'Materialized table of the partition already exists.\n partition: {}, table: {}'
                    .format(partition.name, table))
            else:
                cursor = connection.cursor()

                # create table
                create_query = self.__class__._get_create_query(partition, table)
                debug_logger.debug(
                    'Creating new materialized view for partition mpr.'
                    '\n    partition: {}, view: {}, query: {}'
                    .format(partition.name, table, create_query))

                cursor.execute(create_query)

                # populate just created table with data from virtual table.
                copy_query = '''INSERT INTO {} SELECT * FROM {};'''.format(table, virtual_table)
                debug_logger.debug(
                    'Populating sqlite table with rows from partition mpr.'
                    '\n    partition: {}, view: {}, query: {}'
                    .format(partition.name, table, copy_query))
                cursor.execute(copy_query)

                cursor.close()

        else:
            cursor = connection.cursor()
            view_q = "CREATE VIEW IF NOT EXISTS {} AS SELECT * FROM {} ".format(partition.vid, virtual_table)
            cursor.execute(view_q)
            cursor.close()

        if index_columns is not None:
            self.index(connection,table, index_columns)

        return table