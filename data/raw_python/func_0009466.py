def addFlatField(self, arr, date=None, info='', error=None,
                     light_spectrum='visible'):
        '''
        light_spectrum = light, IR ...
        '''
        self._registerLight(light_spectrum)
        self._checkShape(arr)
        date = _toDate(date)
        f = self.coeffs['flat field']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(_insertDateIndex(date, f[light_spectrum]),
                                 [date, info, arr, error])