def get_slice(self, start_index=None, stop_index=None, columns=None, as_dict=False):
        """
        For sorted DataFrames will return either a DataFrame or dict of all of the rows where the index is greater than
        or equal to the start_index if provided and less than or equal to the stop_index if provided. If either the
        start or stop index is None then will include from the first or last element, similar to standard python
        slide of [:5] or [:5]. Both end points are considered inclusive.
         
        :param start_index: lowest index value to include, or None to start from the first row
        :param stop_index: highest index value to include, or None to end at the last row
        :param columns: list of column names to include, or None for all columns
        :param as_dict: if True then return a tuple of (list of index, dict of column names: list data values)
        :return: DataFrame or tuple
        """
        if not self._sort:
            raise RuntimeError('Can only use get_slice on sorted DataFrames')

        if columns is None:
            columns = self._columns
        elif all([isinstance(i, bool) for i in columns]):
            if len(columns) != len(self._columns):
                raise ValueError('boolean column list must be same size of existing columns')
            columns = list(compress(self._columns, columns))

        start_location = bisect_left(self._index, start_index) if start_index is not None else None
        stop_location = bisect_right(self._index, stop_index) if stop_index is not None else None

        index = self._index[start_location:stop_location]
        data = dict()
        for column in columns:
            c = self._columns.index(column)
            data[column] = self._data[c][start_location:stop_location]

        if as_dict:
            return index, data
        else:
            data = data if data else None  # if the dict is empty, convert to None
            return DataFrame(data=data, index=index, columns=columns, index_name=self._index_name, sort=self._sort,
                             use_blist=self._blist)