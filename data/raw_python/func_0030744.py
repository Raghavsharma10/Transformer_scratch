def copy_dataset(self, ds, incver=False, cb=None, **kwargs):
        """
        Copy a dataset into the database.
        :param ds: The source dataset to copy
        :param cb: A progress callback, taking two parameters: cb(message, num_records)
        :return:
        """
        from ambry.orm import Table, Column, Partition, File, ColumnStat, Code, \
            DataSource, SourceTable, SourceColumn

        tables = [Table, Column, Partition, File, ColumnStat, Code, SourceTable, SourceColumn, DataSource]

        return self._copy_dataset_copy(ds, tables, incver, cb, **kwargs)