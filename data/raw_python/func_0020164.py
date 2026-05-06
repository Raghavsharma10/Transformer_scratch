def plot_pipeline(self, pipeline, *args, **kwargs):
        '''
        Plots the light curve for the target de-trended with a given pipeline.

        :param str pipeline: The name of the pipeline (lowercase). Options \
               are 'everest2', 'everest1', and other mission-specific \
               pipelines. For `K2`, the available pipelines are 'k2sff' \
               and 'k2sc'.

        Additional :py:obj:`args` and :py:obj:`kwargs` are passed directly to
        the :py:func:`pipelines.plot` function of the mission.

        '''

        if pipeline != 'everest2':
            return getattr(missions, self.mission).pipelines.plot(self.ID,
                                                                  pipeline,
                                                                  *args,
                                                                  **kwargs)

        else:

            # We're going to plot the everest 2 light curve like we plot
            # the other pipelines for easy comparison
            plot_raw = kwargs.get('plot_raw', False)
            plot_cbv = kwargs.get('plot_cbv', True)
            show = kwargs.get('show', True)

            if plot_raw:
                y = self.fraw
                ylabel = 'Raw Flux'
            elif plot_cbv:
                y = self.fcor
                ylabel = "EVEREST2 Flux"
            else:
                y = self.flux
                ylabel = "EVEREST2 Flux"

            # Remove nans
            bnmask = np.concatenate([self.nanmask, self.badmask])
            time = np.delete(self.time, bnmask)
            flux = np.delete(y, bnmask)

            # Plot it
            fig, ax = pl.subplots(1, figsize=(10, 4))
            fig.subplots_adjust(bottom=0.15)
            ax.plot(time, flux, "k.", markersize=3, alpha=0.5)

            # Axis limits
            N = int(0.995 * len(flux))
            hi, lo = flux[np.argsort(flux)][[N, -N]]
            pad = (hi - lo) * 0.1
            ylim = (lo - pad, hi + pad)
            ax.set_ylim(ylim)

            # Plot bad data points
            ax.plot(self.time[self.badmask], y[self.badmask],
                    "r.", markersize=3, alpha=0.2)

            # Show the CDPP
            ax.annotate('%.2f ppm' % self._mission.CDPP(flux),
                        xy=(0.98, 0.975), xycoords='axes fraction',
                        ha='right', va='top', fontsize=12, color='r',
                        zorder=99)

            # Appearance
            ax.margins(0, None)
            ax.set_xlabel("Time (%s)" % self._mission.TIMEUNITS, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            fig.canvas.set_window_title("EVEREST2: EPIC %d" % (self.ID))

            if show:
                pl.show()
                pl.close()
            else:
                return fig, ax