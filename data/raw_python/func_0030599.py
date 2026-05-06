def mangle_column_name(self, i, n):
        """
        Override this method to change the way that column names from the source are altered to
        become column names in the schema. This method is called from :py:meth:`mangle_header` for each column in the
        header, and :py:meth:`mangle_header` is called from the RowGenerator, so it will alter the row both when the
        schema is being generated and when data are being inserted into the partition.

        Implement it in your bundle class to change the how columsn are converted from the source into database-friendly
        names

        :param i: Column number
        :param n: Original column name
        :return: A new column name
        """
        from ambry.orm import Column

        if not n:
            return 'column{}'.format(i)

        mn = Column.mangle_name(str(n).strip())

        return mn