def mask_planet(self, t0, period, dur=0.2):
        '''
        Mask all of the transits/eclipses of a given planet/EB. After calling
        this method, you must re-compute the model by calling
        :py:meth:`compute` in order for the mask to take effect.

        :param float t0: The time of first transit (same units as light curve)
        :param float period: The period of the planet in days
        :param foat dur: The transit duration in days. Default 0.2

        '''

        mask = []
        t0 += np.ceil((self.time[0] - dur - t0) / period) * period
        for t in np.arange(t0, self.time[-1] + dur, period):
            mask.extend(np.where(np.abs(self.time - t) < dur / 2.)[0])
        self.transitmask = np.array(
            list(set(np.concatenate([self.transitmask, mask]))))