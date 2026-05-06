def get_category(self, column):
        """Returns categories for the specified numeric values

        Args:
            column(pandas.Series): Values to transform into categories

        Returns:
            pandas.Series
        """
        result = pd.Series(index=column.index)

        for category, stats in self.probability_map.items():
            start, end = stats[0]
            result[(start < column) & (column < end)] = category

        return result