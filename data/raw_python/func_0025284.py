def create_data_item(self, title: str=None) -> DataItem:
        """Create an empty data item in the library.

        :param title: The title of the data item (optional).
        :return: The new :py:class:`nion.swift.Facade.DataItem` object.
        :rtype: :py:class:`nion.swift.Facade.DataItem`

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        data_item = DataItemModule.DataItem()
        data_item.ensure_data_source()
        if title is not None:
            data_item.title = title
        self.__document_model.append_data_item(data_item)
        return DataItem(data_item)