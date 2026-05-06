def get_ylim(self):
        '''
        Computes the ideal y-axis limits for the light curve plot. Attempts to
        set the limits equal to those of the raw light curve, but if more than
        1% of the flux lies either above or below these limits, auto-expands
        to include those points. At the end, adds 5% padding to both the
        top and the bottom.

        '''

        bn = np.array(
            list(set(np.concatenate([self.badmask, self.nanmask]))), dtype=int)
        fraw = np.delete(self.fraw, bn)
        lo, hi = fraw[np.argsort(fraw)][[3, -3]]
        flux = np.delete(self.flux, bn)
        fsort = flux[np.argsort(flux)]
        if fsort[int(0.01 * len(fsort))] < lo:
            lo = fsort[int(0.01 * len(fsort))]
        if fsort[int(0.99 * len(fsort))] > hi:
            hi = fsort[int(0.99 * len(fsort))]
        pad = (hi - lo) * 0.05
        ylim = (lo - pad, hi + pad)
        return ylim