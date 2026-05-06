def set_all_xlims(self, xlim, dx, xscale, fontsize=None):
        """Set limits and ticks for x axis for whole figure.

        This will set x axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            xlim (len-2 list of floats): The limits for the axis.
            dx (float): Amount to increment by between the limits.
            xscale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for x axis tick marks.
                Default is None.

        """
        self._set_all_lims('x', xlim, dx, xscale, fontsize)
        return