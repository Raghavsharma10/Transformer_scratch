def addPSF(self, psf, date=None, info='', light_spectrum='visible'):
        '''
        add a new point spread function
        '''
        self._registerLight(light_spectrum)
        date = _toDate(date)

        f = self.coeffs['psf']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(_insertDateIndex(date, f[light_spectrum]),
                                 [date, info, psf])