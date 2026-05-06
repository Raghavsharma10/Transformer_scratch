def _fit_transform_column(self, table, metadata, transformer_name, table_name):
        """Transform a column from table using transformer and given parameters.

        Args:
            table (pandas.DataFrame): Dataframe containing column to transform.
            metadata (dict): Metadata for given column.
            transformer_name (str): Name of transformer to use on column.
            table_name (str): Name of table in original dataset.

        Returns:
            pandas.DataFrame: Dataframe containing the transformed column. If self.missing=True,
                              it will contain a second column containing 0 and 1 marking if that
                              value was originally null or not.
        """

        column_name = metadata['name']
        content = {}
        columns = []

        if self.missing and table[column_name].isnull().any():
            null_transformer = transformers.NullTransformer(metadata)
            clean_column = null_transformer.fit_transform(table[column_name])
            null_name = '?' + column_name
            columns.append(null_name)
            content[null_name] = clean_column[null_name].values
            table[column_name] = clean_column[column_name]

        transformer_class = self.get_class(transformer_name)
        transformer = transformer_class(metadata)

        self.transformers[(table_name, column_name)] = transformer
        content[column_name] = transformer.fit_transform(table)[column_name].values

        columns = [column_name] + columns
        return pd.DataFrame(content, columns=columns)