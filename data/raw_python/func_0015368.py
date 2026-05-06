def scaled_data(self, data_class, x_range=None, y_range=None):
        """Scale `self.data` as appropriate for the given data encoding
        (data_class) and return it.

        An optional `y_range` -- a 2-tuple (lower, upper) -- can be
        given to specify the y-axis bounds. If not given, the range is
        inferred from the data: (0, <max-value>) presuming no negative
        values, or (<min-value>, <max-value>) if there are negative
        values.  `self.scaled_y_range` is set to the actual lower and
        upper scaling range.

        Ditto for `x_range`. Note that some chart types don't have x-axis
        data.
        """
        self.scaled_data_class = data_class

        # Determine the x-axis range for scaling.
        if x_range is None:
            x_range = self.data_x_range()
            if x_range and x_range[0] > 0:
                x_range = (x_range[0], x_range[1])
        self.scaled_x_range = x_range

        # Determine the y-axis range for scaling.
        if y_range is None:
            y_range = self.data_y_range()
            if y_range and y_range[0] > 0:
                y_range = (y_range[0], y_range[1])
        self.scaled_y_range = y_range

        scaled_data = []
        for type, dataset in self.annotated_data():
            if type == 'x':
                scale_range = x_range
            elif type == 'y':
                scale_range = y_range
            elif type == 'marker-size':
                scale_range = (0, max(dataset))
            scaled_dataset = []
            for v in dataset:
                if v is None:
                    scaled_dataset.append(None)
                else:
                    scaled_dataset.append(
                        data_class.scale_value(v, scale_range))
            scaled_data.append(scaled_dataset)
        return scaled_data