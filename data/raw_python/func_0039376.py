def _fit(self, col):
        """Create a map of the empirical probability for each category.

        Args:
            col(pandas.DataFrame): Data to transform.
        """

        column = col[self.col_name].replace({np.nan: np.inf})
        frequencies = column.groupby(column).count().rename({np.inf: None}).to_dict()
        # next set probability ranges on interval [0,1]
        start = 0
        end = 0
        num_vals = len(col)

        for val in frequencies:
            prob = frequencies[val] / num_vals
            end = start + prob
            interval = (start, end)
            mean = np.mean(interval)
            std = prob / 6
            self.probability_map[val] = (interval, mean, std)
            start = end