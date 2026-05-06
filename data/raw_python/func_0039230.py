def reverse_transform(self, tables, table_metas=None, missing=None):
        """Transform data back to its original format.

        Args:
            tables(dict):   mapping of table names to `tuple` where each tuple is on the form
                            (`pandas.DataFrame`, `dict`). The `DataFrame` contains the transformed
                            data and the `dict` the corresponding meta information.
                            If not specified, the tables will be retrieved using the meta_file.

            table_metas(dict):  Full metadata file for the dataset.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            dict: Map from `str` (table_names) to `pandas.DataFrame` (transformed data).
        """

        if missing is None:
            missing = self.missing

        else:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('reverse_transform'), DeprecationWarning)

        reverse = {}

        for table_name in tables:
            table = tables[table_name]
            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            reverse[table_name] = self.reverse_transform_table(table, table_meta)

        return reverse