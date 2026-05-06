def transform_table(self, table, table_meta, missing=None):
        """Apply the stored transformers to `table`.

        Args:
            table(pandas.DataFrame):     Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            pandas.DataFrame: Transformed table.
        """

        if missing is None:
            missing = self.missing

        else:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('transform_table'), DeprecationWarning)

        content = {}
        columns = []
        table_name = table_meta['name']

        for field in table_meta['fields']:
            column_name = field['name']

            if missing and table[column_name].isnull().any():
                null_transformer = transformers.NullTransformer(field)
                clean_column = null_transformer.fit_transform(table[column_name])
                null_name = '?' + column_name
                columns.append(null_name)
                content[null_name] = clean_column[null_name].values
                column = clean_column[column_name]

            else:
                column = table[column_name].to_frame()

            transformer = self.transformers[(table_name, column_name)]
            content[column_name] = transformer.transform(column)[column_name].values
            columns.append(column_name)

        return pd.DataFrame(content, columns=columns)