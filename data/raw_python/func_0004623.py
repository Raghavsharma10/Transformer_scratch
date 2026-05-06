def _pad_data(self, index_len):
        """
        Pad the data in Series with [None] to ensure that data is the same length as index

        :param index_len: length of index to extend data to
        :return: nothing
        """
        self._data.extend([None] * (index_len - len(self._data)))