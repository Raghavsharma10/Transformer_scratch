def display_xdata(self) -> DataAndMetadata.DataAndMetadata:
        """Return the extended data of this data item display.

        Display data will always be 1d or 2d and either int, float, or RGB data type.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        display_data_channel = self.__display_item.display_data_channel
        return display_data_channel.get_calculated_display_values(True).display_data_and_metadata