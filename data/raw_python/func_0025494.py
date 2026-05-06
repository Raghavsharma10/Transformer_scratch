def get_calculated_display_values(self, immediate: bool=False) -> DisplayValues:
        """Return the display values.

        Return the current (possibly uncalculated) display values unless 'immediate' is specified.

        If 'immediate', return the existing (calculated) values if they exist. Using the 'immediate' values
        avoids calculation except in cases where the display values haven't already been calculated.
        """
        if not immediate or not self.__is_master or not self.__last_display_values:
            if not self.__current_display_values and self.__data_item:
                self.__current_display_values = DisplayValues(self.__data_item.xdata, self.sequence_index, self.collection_index, self.slice_center, self.slice_width, self.display_limits, self.complex_display_type, self.__color_map_data)

                def finalize(display_values):
                    self.__last_display_values = display_values
                    self.display_values_changed_event.fire()

                self.__current_display_values.on_finalize = finalize
            return self.__current_display_values
        return self.__last_display_values