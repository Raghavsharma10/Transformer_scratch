def calculate_y_ticks(self, plot_height):
        """Calculate the y-axis items dependent on the plot height."""

        calibrated_data_min = self.calibrated_data_min
        calibrated_data_max = self.calibrated_data_max
        calibrated_data_range = calibrated_data_max - calibrated_data_min

        ticker = self.y_ticker
        y_ticks = list()
        for tick_value, tick_label in zip(ticker.values, ticker.labels):
            if calibrated_data_range != 0.0:
                y_tick = plot_height - plot_height * (tick_value - calibrated_data_min) / calibrated_data_range
            else:
                y_tick = plot_height - plot_height * 0.5
            if y_tick >= 0 and y_tick <= plot_height:
                y_ticks.append((y_tick, tick_label))

        return y_ticks