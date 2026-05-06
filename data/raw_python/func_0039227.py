def reverse_transform_table(self, table, table_meta, missing=None):
        """Transform a `table` back to its original format.

        Args:
            table(pandas.DataFrame):     Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            pandas.DataFrame: Table in original format.
        """

        if missing is None:
            missing = self.missing

        else:
            self.missing = missing
            warnings.warn(
                DEPRECATION_MESSAGE.format('reverse_transform_table'), DeprecationWarning)

        result = pd.DataFrame(index=table.index)
        table_name = table_meta['name']

        for field in table_meta['fields']:
            new_column = self._reverse_transform_column(table, field, table_name)
            if new_column is not None:
                result[field['name']] = new_column

        return result