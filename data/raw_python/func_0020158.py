def compute(self):
        '''
        Re-compute the :py:mod:`everest` model for the given
        value of :py:obj:`lambda`.
        For long cadence `k2` light curves, this should take several
        seconds. For short cadence `k2` light curves, it may take a
        few minutes. Note that this is a simple wrapper around
        :py:func:`everest.Basecamp.compute`.

        '''

        # If we're doing iterative PLD, get the normalization
        if self.model_name == 'iPLD':
            self._get_norm()

        # Compute as usual
        super(Everest, self).compute()

        # Make NaN cadences NaNs
        self.flux[self.nanmask] = np.nan