def reverse_axis(self, axis_to_reverse):
        """Reverse an axis in all figure plots.

        This will reverse the tick marks on an axis for each plot in the figure.
        It can be overridden in SinglePlot class.

        Args:
            axis_to_reverse (str): Axis to reverse. Supports `x` and `y`.

        Raises:
            ValueError: The string representing the axis to reverse is not `x` or `y`.

        """
        if axis_to_reverse.lower() == 'x':
            self.general.reverse_x_axis = True
        if axis_to_reverse.lower() == 'y':
            self.general.reverse_y_axis = True
        if axis_to_reverse.lower() != 'x' or axis_to_reverse.lower() != 'y':
            raise ValueError('Axis for reversing needs to be either x or y.')
        return