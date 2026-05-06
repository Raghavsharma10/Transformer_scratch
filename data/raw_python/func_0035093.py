def set_ylim(self, xlims, dx, xscale, reverse=False):
        """Set y limits for plot.

        This will set the limits for the y axis
        for the specific plot.

        Args:
            ylims (len-2 list of floats): The limits for the axis.
            dy (float): Amount to increment by between the limits.
            yscale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        self._set_axis_limits('y', xlims, dx, xscale, reverse)
        return