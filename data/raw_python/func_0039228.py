def fit_transform(
            self, tables=None, transformer_dict=None, transformer_list=None, missing=None):
        """Create, apply and store the specified transformers for the given tables.

        Args:
            tables(dict):   Mapping of table names to `tuple` where each tuple is on the form
                            (`pandas.DataFrame`, `dict`). The `DataFrame` contains the table data
                            and the `dict` the corresponding meta information.
                            If not specified, the tables will be retrieved using the meta_file.

            transformer_dict(dict):     Mapping  `tuple(str, str)` -> `str` where the tuple is
                                        (table_name, column_name).

            transformer_list(list):     List of transformers to use. Overrides the transformers in
                                        the meta_file.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            dict: Map from `str` (table_names) to `pandas.DataFrame` (transformed data).
        """

        if missing is None:
            missing = self.missing

        else:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('fit_transform'), DeprecationWarning)

        transformed = {}

        if tables is None:
            tables = self.table_dict

        if transformer_dict is None and transformer_list is None:
            transformer_dict = self.transformer_dict

        for table_name in tables:
            table, table_meta = tables[table_name]
            transformed_table = self.fit_transform_table(
                table, table_meta, transformer_dict, transformer_list)

            transformed[table_name] = transformed_table

        return transformed