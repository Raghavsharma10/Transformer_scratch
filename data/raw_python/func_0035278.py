def set_x_grid_info(self, x_low, x_high, num_x, xscale, xval_name):
        """Set the grid values for x.

        Create information for the grid of x values.

        Args:
            num_x (int): Number of points on axis.
            x_low/x_high (float): Lowest/highest value for the axis.
            xscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            xval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.

        """
        self._set_grid_info('x', x_low, x_high, num_x, xscale, xval_name)
        return