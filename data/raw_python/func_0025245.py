def focused_data_item(self) -> typing.Optional[DataItem.DataItem]:
        """Return the data item with keyboard focus."""
        return self.__focused_display_item.data_item if self.__focused_display_item else None