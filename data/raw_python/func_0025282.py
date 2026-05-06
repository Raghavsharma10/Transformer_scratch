def get_source_data_items(self, data_item: DataItem) -> typing.List[DataItem]:
        """Return the list of data items that are data sources for the data item.

        :return: The list of :py:class:`nion.swift.Facade.DataItem` objects.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [DataItem(data_item) for data_item in self._document_model.get_source_data_items(data_item._data_item)] if data_item else None