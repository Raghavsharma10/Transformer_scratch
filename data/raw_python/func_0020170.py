def plot_folded(self, t0, period, dur=0.2):
        '''
        Plot the light curve folded on a given `period` and centered at `t0`.
        When plotting folded transits, please mask them using
        :py:meth:`mask_planet` and re-compute the model using
        :py:meth:`compute`.

        :param float t0: The time at which to center the plot \
               (same units as light curve)
        :param float period: The period of the folding operation
        :param float dur: The transit duration in days. Default 0.2

        '''

        # Mask the planet
        self.mask_planet(t0, period, dur)

        # Whiten
        gp = GP(self.kernel, self.kernel_params, white=False)
        gp.compute(self.apply_mask(self.time), self.apply_mask(self.fraw_err))
        med = np.nanmedian(self.apply_mask(self.flux))
        y, _ = gp.predict(self.apply_mask(self.flux) - med, self.time)
        fwhite = (self.flux - y)
        fwhite /= np.nanmedian(fwhite)

        # Fold
        tfold = (self.time - t0 - period / 2.) % period - period / 2.

        # Crop
        inds = np.where(np.abs(tfold) < 2 * dur)[0]
        x = tfold[inds]
        y = fwhite[inds]

        # Plot
        fig, ax = pl.subplots(1, figsize=(9, 5))
        fig.subplots_adjust(bottom=0.125)
        ax.plot(x, y, 'k.', alpha=0.5)

        # Get ylims
        yfin = np.delete(y, np.where(np.isnan(y)))
        lo, hi = yfin[np.argsort(yfin)][[3, -3]]
        pad = (hi - lo) * 0.1
        ylim = (lo - pad, hi + pad)
        ax.set_ylim(*ylim)

        # Appearance
        ax.set_xlabel(r'Time (days)', fontsize=18)
        ax.set_ylabel(r'Normalized Flux', fontsize=18)
        fig.canvas.set_window_title(
            '%s %d' % (self._mission.IDSTRING, self.ID))

        pl.show()