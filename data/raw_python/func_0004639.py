def from_series(cls, series, offset=0):
        """
        Creates and return a Series from a Series

        :param series: raccoon Series
        :param offset: offset value must be provided as there is no equivalent for a DataFrame
        :return: Series
        """
        return cls(data=series.data, index=series.index, data_name=series.data_name, index_name=series.index_name,
                   sort=series.sort, offset=offset)