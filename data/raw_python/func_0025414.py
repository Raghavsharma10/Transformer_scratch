def size_to_content(self, get_font_metrics_fn):
        """ Size the canvas item to the proper width, the maximum of any label. """
        new_sizing = self.copy_sizing()

        new_sizing.minimum_width = 0
        new_sizing.maximum_width = 0

        axes = self.__axes
        if axes and axes.is_valid:

            # calculate the width based on the label lengths
            font = "{0:d}px".format(self.font_size)

            max_width = 0
            y_range = axes.calibrated_data_max - axes.calibrated_data_min
            label = axes.y_ticker.value_label(axes.calibrated_data_max + y_range * 5)
            max_width = max(max_width, get_font_metrics_fn(font, label).width)
            label = axes.y_ticker.value_label(axes.calibrated_data_min - y_range * 5)
            max_width = max(max_width, get_font_metrics_fn(font, label).width)

            new_sizing.minimum_width = max_width
            new_sizing.maximum_width = max_width

        self.update_sizing(new_sizing)