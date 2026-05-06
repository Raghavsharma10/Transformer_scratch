def create_data_item_from_data(self, data: numpy.ndarray, title: str=None) -> DataItem:
        """Create a data item in the library from an ndarray.

        The data for the data item will be written to disk immediately and unloaded from memory. If you wish to delay
        writing to disk and keep using the data, create an empty data item and use the data item methods to modify
        the data.

        :param data: The data (ndarray).
        :param title: The title of the data item (optional).
        :return: The new :py:class:`nion.swift.Facade.DataItem` object.
        :rtype: :py:class:`nion.swift.Facade.DataItem`

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return self.create_data_item_from_data_and_metadata(DataAndMetadata.DataAndMetadata.from_data(data), title)