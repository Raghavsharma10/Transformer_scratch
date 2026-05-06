def set_data_item(self, data_item: DataItem) -> None:
        """Set the data item associated with this display panel.

        :param data_item: The :py:class:`nion.swift.Facade.DataItem` object to add.

        This will replace whatever data item, browser, or controller is currently in the display panel with the single
        data item.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        display_panel = self.__display_panel
        if display_panel:
            display_item = data_item._data_item.container.get_display_item_for_data_item(data_item._data_item) if data_item._data_item.container else None
            display_panel.set_display_panel_display_item(display_item)