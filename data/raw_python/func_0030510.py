def install(self, connection, partition, table_name=None, index_columns=None, materialize=False,
                logger=None):
        """ Installs partition's mpr to the database to allow to execute sql queries over mpr.

        Args:
            connection:
            partition (orm.Partition):
            materialize (boolean): if True, create generic table. If False create MED over mpr.

        Returns:
            str: name of the created table.

        """

        raise NotImplementedError