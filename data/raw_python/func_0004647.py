def _pad_data(self, max_len=None):
        """
        Pad the data in DataFrame with [None} to ensure that all columns have the same length.

        :param max_len: If provided will extend all columns to this length, if not then will use the longest column
        :return: nothing
        """
        if not max_len:
            max_len = max([len(x) for x in self._data])
        for _, col in enumerate(self._data):
            col.extend([None] * (max_len - len(col)))