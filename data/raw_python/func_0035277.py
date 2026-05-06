def set_y_grid_info(self, y_low, y_high, num_y, yscale, yval_name):
        """Set the grid values for y.

        Create information for the grid of y values.

        Args:
            num_y (int): Number of points on axis.
            y_low/y_high (float): Lowest/highest value for the axis.
            yscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            yval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.

        """
        self._set_grid_info('y', y_low, y_high, num_y, yscale, yval_name)
        return