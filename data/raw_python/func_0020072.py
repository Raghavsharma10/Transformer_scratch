def get_cdpp_arr(self, flux=None):
        '''
        Returns the CDPP value in *ppm* for each of the
        chunks in the light curve.

        '''

        if flux is None:
            flux = self.flux
        return np.array([self._mission.CDPP(flux[self.get_masked_chunk(b)],
                        cadence=self.cadence)
                        for b, _ in enumerate(self.breakpoints)])