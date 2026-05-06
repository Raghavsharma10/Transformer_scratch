def calculate_x_ticks(self, plot_width):
        """Calculate the x-axis items dependent on the plot width."""

        x_calibration = self.x_calibration

        uncalibrated_data_left = self.__uncalibrated_left_channel
        uncalibrated_data_right = self.__uncalibrated_right_channel

        calibrated_data_left = x_calibration.convert_to_calibrated_value(uncalibrated_data_left) if x_calibration is not None else uncalibrated_data_left
        calibrated_data_right = x_calibration.convert_to_calibrated_value(uncalibrated_data_right) if x_calibration is not None else uncalibrated_data_right
        calibrated_data_left, calibrated_data_right = min(calibrated_data_left, calibrated_data_right), max(calibrated_data_left, calibrated_data_right)

        graph_left, graph_right, tick_values, division, precision = Geometry.make_pretty_range(calibrated_data_left, calibrated_data_right)

        drawn_data_width = self.drawn_right_channel - self.drawn_left_channel

        x_ticks = list()
        if drawn_data_width > 0.0:
            for tick_value in tick_values:
                label = nice_label(tick_value, precision)
                data_tick = x_calibration.convert_from_calibrated_value(tick_value) if x_calibration else tick_value
                x_tick = plot_width * (data_tick - self.drawn_left_channel) / drawn_data_width
                if x_tick >= 0 and x_tick <= plot_width:
                    x_ticks.append((x_tick, label))

        return x_ticks