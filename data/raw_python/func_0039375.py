def anonymize_column(self, col):
        """Map the values of column to new ones of the same type.

        It replaces the values from others generated using `faker`. It will however,
        keep the original distribution. That mean that the generated `probability_map` for both
        will have the same values, but different keys.

        Args:
            col (pandas.DataFrame): Dataframe containing the column to anonymize.

        Returns:
            pd.DataFrame: DataFrame with its values mapped to new ones,
                          keeping the original distribution.

        Raises:
            ValueError: A `ValueError` is raised if faker is not able to provide enought
                        different values.
        """

        column = col[self.col_name]

        generator = self.get_generator()
        original_values = column[~pd.isnull(column)].unique()
        new_values = [generator() for x in range(len(original_values))]

        if len(new_values) != len(set(new_values)):
            raise ValueError(
                'There are not enought different values on faker provider'
                'for category {}'.format(self.category)
            )

        value_map = dict(zip(original_values, new_values))
        column = column.apply(value_map.get)

        return column.to_frame()