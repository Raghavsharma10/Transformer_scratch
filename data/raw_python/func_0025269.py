def data_item(self) -> DataItem:
        """Return the data item associated with this display panel.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        display_panel = self.__display_panel
        if not display_panel:
            return None
        data_item = display_panel.data_item
        return DataItem(data_item) if data_item else None