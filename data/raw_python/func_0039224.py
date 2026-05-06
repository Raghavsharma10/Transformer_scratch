def _reverse_transform_column(self, table, metadata, table_name):
        """Reverses the transformtion on a column from table using the given parameters.

        Args:
            table (pandas.DataFrame): Dataframe containing column to transform.
            metadata (dict): Metadata for given column.
            table_name (str): Name of table in original dataset.

        Returns:
            pandas.DataFrame: Dataframe containing the transformed column. If self.missing=True,
                              it will contain a second column containing 0 and 1 marking if that
                              value was originally null or not.
                              It will return None in the case the column is not in the table.
        """

        column_name = metadata['name']

        if column_name not in table:
            return

        null_name = '?' + column_name
        content = pd.DataFrame(columns=[column_name], index=table.index)
        transformer = self.transformers[(table_name, column_name)]
        content[column_name] = transformer.reverse_transform(table[column_name].to_frame())

        if self.missing and null_name in table[column_name]:
            content[null_name] = table.pop(null_name)
            null_transformer = transformers.NullTransformer(metadata)
            content[column_name] = null_transformer.reverse_transform(content)

        return content