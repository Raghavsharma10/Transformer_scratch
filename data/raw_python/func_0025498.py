def snapshot(self):
        """Return a new library item which is a copy of this one with any dynamic behavior made static."""
        display_item = self.__class__()
        display_item.display_type = self.display_type
        # metadata
        display_item._set_persistent_property_value("title", self._get_persistent_property_value("title"))
        display_item._set_persistent_property_value("caption", self._get_persistent_property_value("caption"))
        display_item._set_persistent_property_value("description", self._get_persistent_property_value("description"))
        display_item._set_persistent_property_value("session_id", self._get_persistent_property_value("session_id"))
        display_item._set_persistent_property_value("calibration_style_id", self._get_persistent_property_value("calibration_style_id"))
        display_item._set_persistent_property_value("display_properties", self._get_persistent_property_value("display_properties"))
        display_item.created = self.created
        for graphic in self.graphics:
            display_item.add_graphic(copy.deepcopy(graphic))
        for display_data_channel in self.display_data_channels:
            display_item.append_display_data_channel(copy.deepcopy(display_data_channel))
        # this goes after the display data channels so that the layers don't get adjusted
        display_item._set_persistent_property_value("display_layers", self._get_persistent_property_value("display_layers"))
        return display_item