def get_slice(self, start_index=None, stop_index=None, as_list=False):
        """
        For sorted Series will return either a Series or list of all of the rows where the index is greater than
        or equal to the start_index if provided and less than or equal to the stop_index if provided. If either the
        start or stop index is None then will include from the first or last element, similar to standard python
        slide of [:5] or [:5]. Both end points are considered inclusive.

        :param start_index: lowest index value to include, or None to start from the first row
        :param stop_index: highest index value to include, or None to end at the last row
        :param as_list: if True then return a list of the indexes and values
        :return: Series or tuple of (index list, values list)
        """
        if not self._sort:
            raise RuntimeError('Can only use get_slice on sorted Series')

        start_location = bisect_left(self._index, start_index) if start_index is not None else None
        stop_location = bisect_right(self._index, stop_index) if stop_index is not None else None

        index = self._index[start_location:stop_location]
        data = self._data[start_location:stop_location]

        if as_list:
            return index, data
        else:
            return Series(data=data, index=index, data_name=self._data_name, index_name=self._index_name,
                          sort=self._sort)