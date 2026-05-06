def _create_index(self, table_name, index_columns):
        """
        Creates an index over multiple columns of a given table.

        Parameters
        ----------
        table_name : str

        index_columns : iterable of str
            Which columns should be indexed
        """

        logger.info(
            "Creating index on %s (%s)",
            table_name,
            ", ".join(index_columns))
        index_name = "%s_index_%s" % (
            table_name,
            "_".join(index_columns))
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS %s ON %s (%s)" % (
                index_name,
                table_name,
                ", ".join(index_columns)))