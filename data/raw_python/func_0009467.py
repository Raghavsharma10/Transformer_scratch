def addLens(self, lens, date=None, info='', light_spectrum='visible'):
        '''
        lens -> instance of LensDistortion or saved file
        '''
        self._registerLight(light_spectrum)
        date = _toDate(date)

        if not isinstance(lens, LensDistortion):
            l = LensDistortion()
            l.readFromFile(lens)
            lens = l

        f = self.coeffs['lens']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(_insertDateIndex(date, f[light_spectrum]),
                                 [date, info, lens.coeffs])