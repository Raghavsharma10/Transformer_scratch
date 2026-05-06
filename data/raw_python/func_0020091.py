def plot_final(self, ax):
        '''
        Plots the final de-trended light curve.

        '''

        # Plot the light curve
        bnmask = np.array(
            list(set(np.concatenate([self.badmask, self.nanmask]))), dtype=int)

        def M(x): return np.delete(x, bnmask)
        if (self.cadence == 'lc') or (len(self.time) < 4000):
            ax.plot(M(self.time), M(self.flux), ls='none',
                    marker='.', color='k', markersize=2, alpha=0.3)
        else:
            ax.plot(M(self.time), M(self.flux), ls='none', marker='.',
                    color='k', markersize=2, alpha=0.03, zorder=-1)
            ax.set_rasterization_zorder(0)
        # Hack: Plot invisible first and last points to ensure
        # the x axis limits are the
        # same in the other plots, where we also plot outliers!
        ax.plot(self.time[0], np.nanmedian(M(self.flux)), marker='.', alpha=0)
        ax.plot(self.time[-1], np.nanmedian(M(self.flux)), marker='.', alpha=0)

        # Plot the GP (long cadence only)
        if self.cadence == 'lc':
            gp = GP(self.kernel, self.kernel_params, white=False)
            gp.compute(self.apply_mask(self.time),
                       self.apply_mask(self.fraw_err))
            med = np.nanmedian(self.apply_mask(self.flux))
            y, _ = gp.predict(self.apply_mask(self.flux) - med, self.time)
            y += med
            ax.plot(M(self.time), M(y), 'r-', lw=0.5, alpha=0.5)

            # Compute the CDPP of the GP-detrended flux
            self.cdppg = self._mission.CDPP(self.apply_mask(
                self.flux - y + med), cadence=self.cadence)

        else:

            # We're not going to calculate this
            self.cdppg = 0.

        # Appearance
        ax.annotate('Final', xy=(0.98, 0.025), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10, alpha=0.5,
                    fontweight='bold')
        ax.margins(0.01, 0.1)

        # Get y lims that bound 99% of the flux
        flux = np.delete(self.flux, bnmask)
        N = int(0.995 * len(flux))
        hi, lo = flux[np.argsort(flux)][[N, -N]]
        fsort = flux[np.argsort(flux)]
        pad = (hi - lo) * 0.1
        ylim = (lo - pad, hi + pad)
        ax.set_ylim(ylim)
        ax.get_yaxis().set_major_formatter(Formatter.Flux)