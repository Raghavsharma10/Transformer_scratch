def update_display_properties(self, display_calibration_info, display_properties: typing.Mapping, display_layers: typing.Sequence[typing.Mapping]) -> None:
        """Update the display values. Called from display panel.

        This method saves the display values and data and triggers an update. It should be as fast as possible.

        As a layer, this canvas item will respond to the update by calling prepare_render on the layer's rendering
        thread. Prepare render will call prepare_display which will construct new axes and update all of the constituent
        canvas items such as the axes labels and the graph layers. Each will trigger its own update if its inputs have
        changed.

        The inefficiencies in this process are that the layer must re-render on each call to this function. There is
        also a cost within the constituent canvas items to check whether the axes or their data has changed.

        When the display is associated with a single data item, the data will be
        """

        # may be called from thread; prevent a race condition with closing.
        with self.__closing_lock:
            if self.__closed:
                return

            displayed_dimensional_scales = display_calibration_info.displayed_dimensional_scales
            displayed_dimensional_calibrations = display_calibration_info.displayed_dimensional_calibrations
            self.__data_scale = displayed_dimensional_scales[-1] if len(displayed_dimensional_scales) > 0 else 1
            self.__displayed_dimensional_calibration = displayed_dimensional_calibrations[-1] if len(displayed_dimensional_calibrations) > 0 else Calibration.Calibration(scale=displayed_dimensional_scales[-1])
            self.__intensity_calibration = display_calibration_info.displayed_intensity_calibration
            self.__calibration_style = display_calibration_info.calibration_style
            self.__y_min = display_properties.get("y_min")
            self.__y_max = display_properties.get("y_max")
            self.__y_style = display_properties.get("y_style", "linear")
            self.__left_channel = display_properties.get("left_channel")
            self.__right_channel = display_properties.get("right_channel")
            self.__legend_position = display_properties.get("legend_position")
            self.__display_layers = display_layers

            if self.__display_values_list and len(self.__display_values_list) > 0:
                self.__xdata_list = [display_values.display_data_and_metadata if display_values else None for display_values in self.__display_values_list]
                xdata0 = self.__xdata_list[0]
                if xdata0:
                    self.__update_frame(xdata0.metadata)
            else:
                self.__xdata_list = list()

            # update the cursor info
            self.__update_cursor_info()

            # mark for update. prepare display will mark children for update if necesssary.
            self.update()