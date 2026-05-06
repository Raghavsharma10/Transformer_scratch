def _set_grid_info(self, which, low, high, num, scale, name):
        """Set the grid values for x or y.

        Create information for the grid of x and y values.

        Args:
            which (str): `x` or `y`.
            low/high (float): Lowest/highest value for the axis.
            num (int): Number of points on axis.
            scale (str): Scale of the axis. Choices are 'log' or 'lin'.
            name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.
            unit (str): Unit for this axis quantity. See GenerateContainer documentation
                for options for the units.

        Raises:
            ValueError: If scale is not 'log' or 'lin'.

        """
        setattr(self.generate_info, which + '_low', low)
        setattr(self.generate_info, which + '_high', high)
        setattr(self.generate_info, 'num_' + which, num)
        setattr(self.generate_info, which + 'val_name', name)

        if scale not in ['lin', 'log']:
            raise ValueError('{} scale must be lin or log.'.format(which))
        setattr(self.generate_info, which + 'scale', scale)
        return