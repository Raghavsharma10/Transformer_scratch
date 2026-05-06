def get_cdpp(self, flux=None):
        '''
        Returns the scalar CDPP for the light curve.

        '''

        if flux is None:
            flux = self.flux
        return self._mission.CDPP(self.apply_mask(flux), cadence=self.cadence)