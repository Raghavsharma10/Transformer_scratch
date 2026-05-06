def plot_aperture(self, show=True):
        '''
        Plot sample postage stamps for the target with the aperture
        outline marked, as well as a high-res target image (if available).

        :param bool show: Show the plot or return the `(fig, ax)` instance? \
               Default :py:obj:`True`

        '''

        # Set up the axes
        fig, ax = pl.subplots(2, 2, figsize=(6, 8))
        fig.subplots_adjust(top=0.975, bottom=0.025, left=0.05,
                            right=0.95, hspace=0.05, wspace=0.05)
        ax = ax.flatten()
        fig.canvas.set_window_title(
            '%s %d' % (self._mission.IDSTRING, self.ID))
        super(Everest, self).plot_aperture(ax, labelsize=12)

        if show:
            pl.show()
            pl.close()
        else:
            return fig, ax