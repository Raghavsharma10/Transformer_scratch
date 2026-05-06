def set_xlim(self, xlims, dx, xscale, reverse=False):
        """Set x limits for plot.

        This will set the limits for the x axis
        for the specific plot.

        Args:
            xlims (len-2 list of floats): The limits for the axis.
            dx (float): Amount to increment by between the limits.
            xscale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        self._set_axis_limits('x', xlims, dx, xscale, reverse)
        return