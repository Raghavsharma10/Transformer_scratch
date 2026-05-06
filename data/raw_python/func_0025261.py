def data(self, data: numpy.ndarray) -> None:
        """Set the data.

        :param data: A numpy ndarray.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        self.__data_item.set_data(numpy.copy(data))