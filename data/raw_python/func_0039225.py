def fit_transform_table(
            self, table, table_meta, transformer_dict=None, transformer_list=None, missing=None):
        """Create, apply and store the specified transformers for `table`.

        Args:
            table(pandas.DataFrame):    Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

            transformer_dict(dict):     Mapping  `tuple(str, str)` -> `str` where the tuple in the
                                        keys represent the (table_name, column_name) and the value
                                        the name of the assigned transformer.

            transformer_list(list):     List of transformers to use. Overrides the transformers in
                                        the meta_file.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            pandas.DataFrame: Transformed table.
        """

        if missing is None:
            missing = self.missing

        else:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('fit_transform_table'), DeprecationWarning)

        result = pd.DataFrame()
        table_name = table_meta['name']

        for field in table_meta['fields']:
            col_name = field['name']

            if transformer_list:
                for transformer_name in transformer_list:
                    if field['type'] == self.get_class(transformer_name).type:
                        transformed = self._fit_transform_column(
                            table, field, transformer_name, table_name)

                        result = pd.concat([result, transformed], axis=1)

            elif (table_name, col_name) in transformer_dict:
                transformer_name = TRANSFORMERS[transformer_dict[(table_name, col_name)]]
                transformed = self._fit_transform_column(
                    table, field, transformer_name, table_name)

                result = pd.concat([result, transformed], axis=1)

        return result