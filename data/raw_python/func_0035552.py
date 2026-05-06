def setup_colorbars(self, plot_call_sign):
        """Setup colorbars for each type of plot.

        Take all of the optional performed during ``__init__`` method and makes the colorbar.

        Args:
            plot_call_sign (obj): Plot instance of ax.contourf with colormapping to
                add as a colorbar.

        """
        self.fig.colorbar(plot_call_sign, cax=self.cbar_ax,
                          ticks=self.cbar_ticks, orientation=self.cbar_orientation)
        # setup colorbar ticks
        (getattr(self.cbar_ax, 'set_' + self.cbar_var + 'ticklabels')
            (self.cbar_tick_labels, fontsize=self.cbar_ticks_fontsize))
        (getattr(self.cbar_ax, 'set_' + self.cbar_var + 'label')
            (self.cbar_label, fontsize=self.cbar_label_fontsize, labelpad=self.cbar_label_pad))

        return