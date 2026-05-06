def append(self, data_frame):
        """
        Append another DataFrame to this DataFrame. If the new data_frame has columns that are not in the current
        DataFrame then new columns will be created. All of the indexes in the data_frame must be different from the
        current indexes or will raise an error.

        :param data_frame: DataFrame to append
        :return: nothing
        """
        if len(data_frame) == 0:  # empty DataFrame, do nothing
            return
        data_frame_index = data_frame.index
        combined_index = self._index + data_frame_index
        if len(set(combined_index)) != len(combined_index):
            raise ValueError('duplicate indexes in DataFrames')

        for c, column in enumerate(data_frame.columns):
            if PYTHON3:
                self.set(indexes=data_frame_index, columns=column, values=data_frame.data[c].copy())
            else:
                self.set(indexes=data_frame_index, columns=column, values=data_frame.data[c][:])