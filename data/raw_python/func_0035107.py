def set_all_ylims(self, ylim, dy, yscale, fontsize=None):
        """Set limits and ticks for y axis for whole figure.

        This will set y axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            ylim (len-2 list of floats): The limits for the axis.
            dy (float): Amount to increment by between the limits.
            yscale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for y axis tick marks.
                Default is None.

        """
        self._set_all_lims('y', ylim, dy, yscale, fontsize)
        return