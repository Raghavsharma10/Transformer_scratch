def from_dataframe(cls, dataframe, column, offset=0):
        """
        Creates and return a Series from a DataFrame and specific column

        :param dataframe: raccoon DataFrame
        :param column: column name
        :param offset: offset value must be provided as there is no equivalent for a DataFrame
        :return: Series
        """
        return cls(data=dataframe.get_entire_column(column, as_list=True), index=dataframe.index,
                   data_name=column, index_name=dataframe.index_name, sort=dataframe.sort, offset=offset)