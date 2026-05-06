def prepare_display(self):
        """Prepare the display.

        This method gets called by the canvas layout/draw engine after being triggered by a call to `update`.

        When data or display parameters change, the internal state of the line plot gets updated. This method takes
        that internal state and updates the child canvas items.

        This method is always run on a thread and should be fast but doesn't need to be instant.
        """
        displayed_dimensional_calibration = self.__displayed_dimensional_calibration
        intensity_calibration = self.__intensity_calibration
        calibration_style = self.__calibration_style
        y_min = self.__y_min
        y_max = self.__y_max
        y_style = self.__y_style
        left_channel = self.__left_channel
        right_channel = self.__right_channel

        scalar_xdata_list = None

        def calculate_scalar_xdata(xdata_list):
            scalar_xdata_list = list()
            for xdata in xdata_list:
                if xdata:
                    scalar_data = Image.scalar_from_array(xdata.data)
                    scalar_data = Image.convert_to_grayscale(scalar_data)
                    scalar_intensity_calibration = calibration_style.get_intensity_calibration(xdata)
                    scalar_dimensional_calibrations = calibration_style.get_dimensional_calibrations(xdata.dimensional_shape, xdata.dimensional_calibrations)
                    if displayed_dimensional_calibration.units == scalar_dimensional_calibrations[-1].units and intensity_calibration.units == scalar_intensity_calibration.units:
                        # the data needs to have an intensity scale matching intensity_calibration. convert the data to use the common scale.
                        scale = scalar_intensity_calibration.scale / intensity_calibration.scale
                        offset = (scalar_intensity_calibration.offset - intensity_calibration.offset) / intensity_calibration.scale
                        scalar_data = scalar_data * scale + offset
                        scalar_xdata_list.append(DataAndMetadata.new_data_and_metadata(scalar_data, scalar_intensity_calibration, scalar_dimensional_calibrations))
                else:
                    scalar_xdata_list.append(None)
            return scalar_xdata_list

        data_scale = self.__data_scale
        xdata_list = self.__xdata_list

        if data_scale is not None:
            # update the line graph data
            left_channel = left_channel if left_channel is not None else 0
            right_channel = right_channel if right_channel is not None else data_scale
            left_channel, right_channel = min(left_channel, right_channel), max(left_channel, right_channel)

            scalar_data_list = None
            if y_min is None or y_max is None and len(xdata_list) > 0:
                scalar_xdata_list = calculate_scalar_xdata(xdata_list)
                scalar_data_list = [xdata.data if xdata else None for xdata in scalar_xdata_list]
            calibrated_data_min, calibrated_data_max, y_ticker = LineGraphCanvasItem.calculate_y_axis(scalar_data_list, y_min, y_max, intensity_calibration, y_style)
            axes = LineGraphCanvasItem.LineGraphAxes(data_scale, calibrated_data_min, calibrated_data_max, left_channel, right_channel, displayed_dimensional_calibration, intensity_calibration, y_style, y_ticker)

            if scalar_xdata_list is None:
                if len(xdata_list) > 0:
                    scalar_xdata_list = calculate_scalar_xdata(xdata_list)
                else:
                    scalar_xdata_list = list()

            if self.__display_frame_rate_id:
                Utility.fps_tick("prepare_"+self.__display_frame_rate_id)

            colors = ('#1E90FF', "#F00", "#0F0", "#00F", "#FF0", "#0FF", "#F0F", "#888", "#800", "#080", "#008", "#CCC", "#880", "#088", "#808", "#964B00")

            display_layers = self.__display_layers

            if len(display_layers) == 0:
                index = 0
                for scalar_index, scalar_xdata in enumerate(scalar_xdata_list):
                    if scalar_xdata and scalar_xdata.is_data_1d:
                        if index < 16:
                            display_layers.append({"fill_color": colors[index] if index == 0 else None, "stroke_color": colors[index] if index > 0 else None, "data_index": scalar_index})
                            index += 1
                    if scalar_xdata and scalar_xdata.is_data_2d:
                        for row in range(min(scalar_xdata.data_shape[-1], 16)):
                            if index < 16:
                                display_layers.append({"fill_color": colors[index] if index == 0 else None, "stroke_color": colors[index] if index > 0 else None, "data_index": scalar_index, "data_row": row})
                                index += 1

            display_layer_count = len(display_layers)

            self.___has_valid_drawn_graph_data = False

            for index, display_layer in enumerate(display_layers):
                if index < 16:
                    fill_color = display_layer.get("fill_color")
                    stroke_color = display_layer.get("stroke_color")
                    data_index = display_layer.get("data_index", 0)
                    data_row = display_layer.get("data_row", 0)
                    if 0 <= data_index < len(scalar_xdata_list):
                        scalar_xdata = scalar_xdata_list[data_index]
                        if scalar_xdata:
                            data_row = max(0, min(scalar_xdata.dimensional_shape[0] - 1, data_row))
                            intensity_calibration = scalar_xdata.intensity_calibration
                            displayed_dimensional_calibration = scalar_xdata.dimensional_calibrations[-1]
                            if scalar_xdata.is_data_2d:
                                scalar_data = scalar_xdata.data[data_row:data_row + 1, :].reshape((scalar_xdata.dimensional_shape[-1],))
                                scalar_xdata = DataAndMetadata.new_data_and_metadata(scalar_data, intensity_calibration, [displayed_dimensional_calibration])
                        line_graph_canvas_item = self.__line_graph_stack.canvas_items[display_layer_count - (index + 1)]
                        line_graph_canvas_item.set_fill_color(fill_color)
                        line_graph_canvas_item.set_stroke_color(stroke_color)
                        line_graph_canvas_item.set_axes(axes)
                        line_graph_canvas_item.set_uncalibrated_xdata(scalar_xdata)
                        self.___has_valid_drawn_graph_data = scalar_xdata is not None

            for index in range(len(display_layers), 16):
                line_graph_canvas_item = self.__line_graph_stack.canvas_items[index]
                line_graph_canvas_item.set_axes(None)
                line_graph_canvas_item.set_uncalibrated_xdata(None)

            legend_position = self.__legend_position
            LegendEntry = collections.namedtuple("LegendEntry", ["label", "fill_color", "stroke_color"])
            legend_entries = list()
            for index, display_layer in enumerate(self.__display_layers):
                data_index = display_layer.get("data_index", None)
                data_row = display_layer.get("data_row", None)
                label = display_layer.get("label", str())
                if not label:
                    if data_index is not None and data_row is not None:
                        label = "Data {}:{}".format(data_index, data_row)
                    elif data_index is not None:
                        label = "Data {}".format(data_index)
                    else:
                        label = "Unknown"
                fill_color = display_layer.get("fill_color")
                stroke_color = display_layer.get("stroke_color")
                legend_entries.append(LegendEntry(label, fill_color, stroke_color))

            self.__update_canvas_items(axes, legend_position, legend_entries)
        else:
            for line_graph_canvas_item in self.__line_graph_stack.canvas_items:
                line_graph_canvas_item.set_axes(None)
                line_graph_canvas_item.set_uncalibrated_xdata(None)
            self.__line_graph_xdata_list = list()
            self.__update_canvas_items(LineGraphCanvasItem.LineGraphAxes(), None, None)