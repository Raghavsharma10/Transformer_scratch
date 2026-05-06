def add_data_item(self, data_item: DataItem) -> None:
        """Add a data item to the group.

        :param data_item: The :py:class:`nion.swift.Facade.DataItem` object to add.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        display_item = data_item._data_item.container.get_display_item_for_data_item(data_item._data_item) if data_item._data_item.container else None
        if display_item:
            self.__data_group.append_display_item(display_item)