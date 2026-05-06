def get_location(self, location, columns=None, as_dict=False, index=True):
        """
        For an index location and either (1) list of columns return a DataFrame or dictionary of the values or
        (2) single column name and return the value of that cell. This is optimized for speed because it does not need
        to lookup the index location with a search. Also can accept relative indexing from the end of the DataFrame
        in standard python notation [-3, -2, -1]
        
        :param location: index location in standard python form of positive or negative number
        :param columns: list of columns, single column name, or None to include all columns
        :param as_dict: if True then return a dictionary
        :param index: if True then include the index in the dictionary if as_dict=True
        :return: DataFrame or dictionary if columns is a list or value if columns is a single column name
        """
        if columns is None:
            columns = self._columns
        elif not isinstance(columns, list):  # single value for columns
            c = self._columns.index(columns)
            return self._data[c][location]
        elif all([isinstance(i, bool) for i in columns]):
            if len(columns) != len(self._columns):
                raise ValueError('boolean column list must be same size of existing columns')
            columns = list(compress(self._columns, columns))
        data = dict()
        for column in columns:
            c = self._columns.index(column)
            data[column] = self._data[c][location]
        index_value = self._index[location]
        if as_dict:
            if index:
                data[self._index_name] = index_value
            return data
        else:
            data = {k: [data[k]] for k in data}  # this makes the dict items lists
            return DataFrame(data=data, index=[index_value], columns=columns, index_name=self._index_name,
                             sort=self._sort)