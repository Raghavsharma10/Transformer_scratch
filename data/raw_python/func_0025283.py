def get_dependent_data_items(self, data_item: DataItem) -> typing.List[DataItem]:
        """Return the dependent data items the data item argument.

        :return: The list of :py:class:`nion.swift.Facade.DataItem` objects.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [DataItem(data_item) for data_item in self._document_model.get_dependent_data_items(data_item._data_item)] if data_item else None