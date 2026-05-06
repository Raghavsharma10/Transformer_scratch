def plot_cbv(self, ax, flux, info, show_cbv=False):
        '''
        Plots the final CBV-corrected light curve.

        '''

        # Plot the light curve
        bnmask = np.array(
            list(set(np.concatenate([self.badmask, self.nanmask]))), dtype=int)

        def M(x): return np.delete(x, bnmask)
        if self.cadence == 'lc':
            ax.plot(M(self.time), M(flux), ls='none', marker='.',
                    color='k', markersize=2, alpha=0.45)
        else:
            ax.plot(M(self.time), M(flux), ls='none', marker='.',
                    color='k', markersize=2, alpha=0.03, zorder=-1)
            ax.set_rasterization_zorder(0)
        # Hack: Plot invisible first and last points to ensure
        # the x axis limits are the
        # same in the other plots, where we also plot outliers!
        ax.plot(self.time[0], np.nanmedian(M(flux)), marker='.', alpha=0)
        ax.plot(self.time[-1], np.nanmedian(M(flux)), marker='.', alpha=0)

        # Show CBV fit?
        if show_cbv:
            ax.plot(self.time, self._mission.FitCBVs(
                self) + np.nanmedian(flux), 'r-', alpha=0.2)

        # Appearance
        ax.annotate(info, xy=(0.98, 0.025), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10, alpha=0.5,
                    fontweight='bold')
        ax.margins(0.01, 0.1)

        # Get y lims that bound 99% of the flux
        flux = np.delete(flux, bnmask)
        N = int(0.995 * len(flux))
        hi, lo = flux[np.argsort(flux)][[N, -N]]
        fsort = flux[np.argsort(flux)]
        pad = (hi - lo) * 0.2
        ylim = (lo - pad, hi + pad)
        ax.set_ylim(ylim)
        ax.get_yaxis().set_major_formatter(Formatter.Flux)
        ax.set_xlabel(r'Time (%s)' % self._mission.TIMEUNITS, fontsize=9)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(7)