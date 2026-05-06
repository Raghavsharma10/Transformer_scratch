def _set_all_lims(self, which, lim, d, scale, fontsize=None):
        """Set limits and ticks for an axis for whole figure.

        This will set axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            lim (len-2 list of floats): The limits for the axis.
            d (float): Amount to increment by between the limits.
            scale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for associated axis tick marks.
                Default is None.

        """

        setattr(self.general, which + 'lims', lim)
        setattr(self.general, 'd' + which, d)
        setattr(self.general, which + 'scale', scale)

        if fontsize is not None:
            setattr(self.general, which + '_tick_label_fontsize', fontsize)
        return